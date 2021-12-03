
from __future__ import annotations
import math
import warnings
import numbers
from collections import namedtuple
from typing import Dict, Any, Optional, List, Tuple, Union
import tensorflow as tf
from returnn.tf.layers.basic import LayerBase, SubnetworkLayer
from returnn.tf.layers.rec import RecLayer
from .module import Module
from ..parameter import Parameter
from ...tensor import Tensor
from .. import init
from ....naming import Naming, TensorEntry


PackedSequence_ = namedtuple(
  'PackedSequence',
  ['data', 'batch_sizes', 'sorted_indices', 'unsorted_indices'],
  defaults=[None] * 4)


class PackedSequence(PackedSequence_):
  """
  Wrapper for torch's PackedSequence

  We connect each PackedSequence with a batched RETURNN tensor. This currently contains some assumptions which might not
  always be satisfied.
  """

  def get_padded_tensor(self, batch_first: bool) -> Tensor:
    from .shape import FlattenBatch
    naming = Naming.get_instance()
    tensor_entry = naming.tensors[self.data]
    assert isinstance(tensor_entry, TensorEntry)
    for call_entry in tensor_entry.output_from_calls:
      if isinstance(call_entry.module.module, FlattenBatch):
        out_tensor = call_entry.orig_inputs_args[0]
        if batch_first != out_tensor.shape[0].is_batch_dim:
          # batch_first refers to whether the output should be batch major or not. Since the input to the PackedSequence
          # was not batch major in this case, we need to transpose.
          out_tensor = out_tensor.transpose(0, 1)
        assert batch_first == out_tensor.shape[0].is_batch_dim
        return out_tensor
    from .shape import UnflattenBatch
    return UnflattenBatch(batch_first=batch_first)(self.data)


def apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
  return tensor.index_select(dim, permutation)


class RNNBase(Module):

  def __init__(self, mode: str, input_size: int, hidden_size: int,
               num_layers: int = 1, bias: bool = True, batch_first: bool = False,
               dropout: float = 0., bidirectional: bool = False) -> None:
    super(RNNBase, self).__init__()
    self.mode = mode
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bias = bias
    self.batch_first = batch_first
    self.dropout = float(dropout)
    self.bidirectional = bidirectional
    num_directions = 2 if bidirectional else 1

    if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or isinstance(dropout, bool):
      raise ValueError("dropout should be a number in range [0, 1] "
                       "representing the probability of an element being "
                       "zeroed")
    if dropout > 0 and num_layers == 1:
      warnings.warn("dropout option adds dropout after all but last "
                    "recurrent layer, so non-zero dropout expects "
                    "num_layers greater than 1, but got dropout={} and "
                    "num_layers={}".format(dropout, num_layers))

    if mode == 'LSTM':
      gate_size = 4 * hidden_size
    elif mode == 'GRU':
      gate_size = 3 * hidden_size
    elif mode == 'RNN_TANH':
      gate_size = hidden_size
    elif mode == 'RNN_RELU':
      gate_size = hidden_size
    else:
      raise ValueError("Unrecognized RNN mode: " + mode)

    self._all_weights = []
    for layer in range(num_layers):
      for direction in range(num_directions):
        layer_input_size = input_size if layer == 0 else hidden_size * num_directions

        w_ih = Parameter(Tensor(gate_size, layer_input_size))
        w_hh = Parameter(Tensor(gate_size, hidden_size))
        b_ih = Parameter(Tensor(gate_size))
        # Second bias vector included for CuDNN compatibility. Only one
        # bias vector is needed in standard definition.
        b_hh = Parameter(Tensor(gate_size))
        layer_params = (w_ih, w_hh, b_ih, b_hh)

        suffix = '_reverse' if direction == 1 else ''
        param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
        if bias:
          param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
        param_names = [x.format(layer, suffix) for x in param_names]

        for name, param in zip(param_names, layer_params):
          setattr(self, name, param)
        self._all_weights.append(param_names)

    self.reset_parameters()

  def flatten_parameters(self) -> None:
    pass

  def reset_parameters(self) -> None:
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      init.uniform_(weight, -stdv, stdv)

  def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]) -> None:
    expected_input_dim = 2 if batch_sizes is not None else 3
    if input.dim() != expected_input_dim:
      raise RuntimeError(
        'input must have {} dimensions, got {}'.format(
          expected_input_dim, input.dim()))
    if self.input_size != input.size(-1):
      raise RuntimeError(
        'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
          self.input_size, input.size(-1)))

  def get_expected_hidden_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
    if batch_sizes is not None:
      mini_batch = batch_sizes[0]
      mini_batch = int(mini_batch)
    else:
      mini_batch = input.size(0) if self.batch_first else input.size(1)
    num_directions = 2 if self.bidirectional else 1
    expected_hidden_size = (self.num_layers * num_directions,
                            mini_batch, self.hidden_size)
    return expected_hidden_size

  def check_hidden_size(self, hx: Tensor, expected_hidden_size: Tuple[int, int, int],
                        msg: str = 'Expected hidden size {}, got {}') -> None:
    if hx.size() != expected_hidden_size:
      raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))

  def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]):
    self.check_input(input, batch_sizes)
    expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

    self.check_hidden_size(hidden, expected_hidden_size)

  def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]):
    if permutation is None:
      return hx
    return apply_permutation(hx, permutation)

  def create_returnn_layer_dict(self, input: Tensor, hx: Optional[Tensor] = None) -> Dict[str, Any]:
    assert self.num_layers == 1  # not implemented otherwise
    return {
      "class": "rec", "unit": self.mode, "from": self._get_input_layer_name(input), "n_out": self.hidden_size}

  def import_params_torch_to_returnn(self, *, layer: LayerBase, torch_module):
    pass  # TODO...

  def __setstate__(self, d):
    super(RNNBase, self).__setstate__(d)
    if 'all_weights' in d:
      self._all_weights = d['all_weights']

    if isinstance(self._all_weights[0][0], str):
      return
    num_layers = self.num_layers
    num_directions = 2 if self.bidirectional else 1
    self._all_weights = []
    for layer in range(num_layers):
      for direction in range(num_directions):
        suffix = '_reverse' if direction == 1 else ''
        weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
        weights = [x.format(layer, suffix) for x in weights]
        if self.bias:
          self._all_weights += [weights]
        else:
          self._all_weights += [weights[:2]]

  @property
  def all_weights(self) -> List[Parameter]:
    return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]


class LSTM(RNNBase):
  def __init__(self, *args, **kwargs):
    super(LSTM, self).__init__('LSTM', *args, **kwargs)

  def check_forward_args(self, input: Tensor, hidden: Tuple[Tensor, Tensor], batch_sizes: Optional[Tensor]):
    self.check_input(input, batch_sizes)
    expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

    self.check_hidden_size(hidden[0], expected_hidden_size,
                           'Expected hidden[0] size {}, got {}')
    self.check_hidden_size(hidden[1], expected_hidden_size,
                           'Expected hidden[1] size {}, got {}')

  def permute_hidden(self, hx: Tuple[Tensor, Tensor], permutation: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    if permutation is None:
      return hx
    return apply_permutation(hx[0], permutation), apply_permutation(hx[1], permutation)

  def create_returnn_layer_dict(self, input: Tensor, hx: Optional[Tensor] = None) -> Dict[str, Any]:
    if isinstance(input, PackedSequence):
      input = input.get_padded_tensor(batch_first=False)
    if not self.bidirectional and self.num_layers == 1:
      d = {
        "class": "rec", "unit": "nativelstm2", "from": self._get_input_layer_name(input),
        "n_out": self.hidden_size}
      if hx is not None:
        assert isinstance(hx, (list, tuple)) and len(hx) == 2
        h_0, c_0 = hx  # h_0/c_0 of shape (num_layers * num_directions, batch, hidden_size)
        h_, c_ = h_0[0], c_0[0]  # (batch,hidden) each
        d["initial_state"] = {
          "h": self._get_input_layer_name(h_),
          "c": self._get_input_layer_name(c_)}
    else:
      input_layer_name = self._get_input_layer_name(input)  # call now, to have nicer order of "data"
      subnet_dict = {}
      for i in range(self.num_layers):
        if not self.bidirectional:
          layer_dict = {
            "class": "rec", "unit": "nativelstm2", "from": "data" if i == 0 else f"layer{i - 1}",
            "n_out": self.hidden_size}
          subnet_dict[f"layer{i}"] = layer_dict
        else:
          layer_dict = {
            "class": "rec", "unit": "nativelstm2", "direction": 1,
            "from": "data" if i == 0 else [f"layer{i - 1}_fwd", f"layer{i - 1}_bwd"], "n_out": self.hidden_size}
          subnet_dict[f"layer{i}_fwd"] = layer_dict.copy()
          layer_dict["direction"] = -1
          subnet_dict[f"layer{i}_bwd"] = layer_dict.copy()
        if hx is not None:
          assert isinstance(hx, (list, tuple)) and len(hx) == 2
          h_0, c_0 = hx  # h_0/c_0 of shape (num_layers * num_directions, batch, hidden_size)
          h_, c_ = h_0[i], c_0[i]  # (batch,hidden) each
          layer_dict["initial_state"] = {
            "h": f"base:{self._get_input_layer_name(h_)}",
            "c": f"base:{self._get_input_layer_name(c_)}"}
      subnet_dict["output"] = {
        "class": "copy", "from": f"layer{self.num_layers - 1}" if not self.bidirectional else [
          f"layer{self.num_layers - 1}_fwd", f"layer{self.num_layers - 1}_bwd"
        ]}
      return {
        "class": "subnetwork", "from": input_layer_name, "subnetwork": subnet_dict}
    return d

  def _get_output_shape_from_returnn(self,
                                     inputs_flat: List[Tensor], layer: LayerBase
                                     ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    naming = Naming.get_instance()
    call_entry = naming.module_call_stack[-1]
    assert call_entry.module.module is self
    orig_input = call_entry.orig_inputs_args[0]
    if isinstance(orig_input, PackedSequence):
      orig_input = orig_input.get_padded_tensor(batch_first=False)
    return self._base_get_output_shape_from_returnn(inputs_flat=[orig_input], layer=layer)

  def make_structured_returnn_output(self, output: Tensor, input: Tensor, hx: Optional[Tensor] = None
                                     ) -> Union[Tensor, Tuple[Tensor], Any]:
    hs, cs = [], []
    dim = output.shape[-1]
    if not self.bidirectional and self.num_layers == 1:
      hs.append(GetLastHiddenState(dim=dim, key="h")(output))
      cs.append(GetLastHiddenState(dim=dim, key="c")(output))
    else:
      for i in range(self.num_layers):
        if not self.bidirectional:
          hs.append(GetLastHiddenState(dim=dim, sub_layer=f"layer{i}", key="h")(output))
          cs.append(GetLastHiddenState(dim=dim, sub_layer=f"layer{i}", key="c")(output))
        else:
          hs.append(GetLastHiddenState(dim=dim // 2, sub_layer=f"layer{i}_fwd", key="h")(output))
          hs.append(GetLastHiddenState(dim=dim // 2, sub_layer=f"layer{i}_bwd", key="h")(output))
          cs.append(GetLastHiddenState(dim=dim // 2, sub_layer=f"layer{i}_fwd", key="c")(output))
          cs.append(GetLastHiddenState(dim=dim // 2, sub_layer=f"layer{i}_bwd", key="c")(output))
    from .operator import Stack
    h = Stack(dim=0)(*hs)
    c = Stack(dim=0)(*cs)
    if isinstance(input, PackedSequence):
      from ..utils.rnn import pack_padded_sequence_with_batch_sizes
      output = pack_padded_sequence_with_batch_sizes(output, input.batch_sizes)
    return output, (h, c)

  def check_returnn_layer(self, layer: LayerBase):
    if not self.bidirectional and self.num_layers == 1:
      assert isinstance(layer, RecLayer)
      assert layer.input_data.dim == self.input_size
    else:
      assert isinstance(layer, SubnetworkLayer)
      assert layer.subnetwork_.construct_layer("data").output.dim == self.input_size

  def import_params_torch_to_returnn(self, *, layer: LayerBase, torch_module: LSTM):
    import torch
    import numpy
    session = tf.compat.v1.get_default_session()
    for i in range(self.num_layers):
      for layer_suffix in ["_fwd", "_bwd"] if self.bidirectional else [""]:
        if not self.bidirectional and self.num_layers == 1:
          assert isinstance(layer, RecLayer)
          sub_layer = layer
        else:
          assert isinstance(layer, SubnetworkLayer)
          sub_layer = layer.subnetwork.layers[f"layer{i}{layer_suffix}"]
          assert isinstance(sub_layer, RecLayer)
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        # RETURNN NativeLstm2 H order: j, i, f, o
        # Torch H order: i, f, j, o  ("W_ii|W_if|W_ig|W_io" in their docs, g==j)
        parameter_suffix = "_reverse" if layer_suffix == "_bwd" else ""
        weight_hh = getattr(torch_module, f"weight_hh_l{i}{parameter_suffix}")  # (H, out)
        weight_ih = getattr(torch_module, f"weight_ih_l{i}{parameter_suffix}")  # (H, in)
        assert isinstance(weight_hh, torch.nn.Parameter)
        assert isinstance(weight_ih, torch.nn.Parameter)
        bias_np = None
        if self.bias:
          bias_hh = getattr(torch_module, f"bias_hh_l{i}{parameter_suffix}")  # (H,)
          bias_ih = getattr(torch_module, f"bias_ih_l{i}{parameter_suffix}")  # (H,)
          assert isinstance(bias_hh, torch.nn.Parameter)
          assert isinstance(bias_ih, torch.nn.Parameter)
          bias_np = bias_hh.detach().cpu().numpy() + bias_ih.detach().cpu().numpy()  # RETURNN has single bias
        weight_hh_np = weight_hh.detach().cpu().numpy().transpose()  # (out, H)
        weight_ih_np = weight_ih.detach().cpu().numpy().transpose()  # (in, H)

        def _torch_to_returnn(x: numpy.ndarray, axis: int) -> numpy.ndarray:
          x_ = numpy.split(x, 4, axis=axis)  # (i,f,j,o)
          y = numpy.concatenate([x_[2], x_[0], x_[1], x_[3]], axis=axis)  # -> (j,i,f,o)
          return y

        if self.bias:
          bias_np = _torch_to_returnn(bias_np, axis=0)
        weight_hh_np = _torch_to_returnn(weight_hh_np, axis=1)
        weight_ih_np = _torch_to_returnn(weight_ih_np, axis=1)

        # RETURNN sub_layer.params: W (in, X), W_re (out, X), b (X,)
        sub_layer.params["W"].load(weight_ih_np, session=session)
        sub_layer.params["W_re"].load(weight_hh_np, session=session)
        if self.bias:
          sub_layer.params["b"].load(bias_np, session=session)


class GetLastHiddenState(Module):
  is_original_torch_module = False

  def __init__(self, *, dim: int, sub_layer: Optional[str] = None, key: Optional[str] = None):
    super(GetLastHiddenState, self).__init__()
    self.sub_layer = sub_layer
    self.key = key
    self.dim = dim

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    layer = self._get_input_layer_name(input)
    if self.sub_layer:
      layer = f"{layer}/{self.sub_layer}"
    d = {"class": "get_last_hidden_state", "from": layer, "n_out": self.dim}
    if self.key:
      d["key"] = self.key
    return d


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
