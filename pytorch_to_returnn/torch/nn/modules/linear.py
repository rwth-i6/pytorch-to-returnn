
from __future__ import annotations
import math
from typing import Dict, Any, Optional
import tensorflow as tf
from returnn.tf.layers.basic import LinearLayer
from .module import Module
from ..parameter import Parameter
from ...tensor import Tensor
from .. import init


class Identity(Module):
  def __init__(self, *args, **kwargs):
    super(Identity, self).__init__()

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    return {"class": "copy", "from": self._get_input_layer_name(input)}


class Linear(Module):
  in_features: int
  out_features: int
  weight: Tensor

  def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
    super(Linear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = Parameter(Tensor(out_features, in_features))
    if bias:
      self.bias = Parameter(Tensor(out_features))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      bound = 1 / math.sqrt(fan_in)
      init.uniform_(self.bias, -bound, bound)

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    return {
      "class": "linear", "from": self._get_input_layer_name(input),
      "n_out": self.out_features,
      "with_bias": self.bias is not None,
      "activation": None}

  def check_returnn_layer(self, layer: LinearLayer):
    assert layer.input_data.dim == self.in_features

  def import_params_torch_to_returnn(self, *, layer: LinearLayer, torch_module: Linear):
    session = tf.compat.v1.get_default_session()
    values = torch_module.weight.detach().numpy()
    values = values.transpose()
    layer.params["W"].load(values, session=session)
    if self.bias is not None:
      layer.params["b"].load(torch_module.bias.detach().numpy(), session=session)


class FunctionalLinear(Module):
  is_original_torch_module = False

  def create_returnn_layer_dict(self, input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Dict[str, Any]:
    assert len(weight.shape) == 2
    out_features, in_features = weight.shape
    if bias is not None:
      assert bias.shape == (out_features,)
    return {
      "class": "linear", "from": self._get_input_layer_name(input),
      "n_out": out_features,
      "with_bias": bias is not None,
      "bias": self._get_input_layer_name(bias) if bias is not None else None,
      "weights": self._get_input_layer_name(weight),
      "use_transposed_weights": True,
      "activation": None}


class Matmul(Module):
  """
  Maps to RETURNN DotLayer
  """
  is_original_torch_module = False

  def create_returnn_layer_dict(self, *inputs: Tensor, **kwargs) -> Dict[str, Any]:
    sources = [self._get_input_layer_name(source) for source in inputs]
    assert len(sources) == 2

    assert len(inputs[0].shape) >= 2, "not implemented otherwise"
    assert len(inputs[1].shape) >= 2, "not implemented otherwise"
    red1 = inputs[0].returnn_naming_entry.get_returnn_axis_description(-1)
    red2 = inputs[1].returnn_naming_entry.get_returnn_axis_description(-2)
    var1 = [inputs[0].returnn_naming_entry.get_returnn_axis_description(-2)]
    var2 = [inputs[1].returnn_naming_entry.get_returnn_axis_description(-1)]

    max_len = max(len(inputs[0].shape), len(inputs[1].shape))
    shape1 = [None] * (max_len - len(inputs[0].shape)) + list(inputs[0].shape)
    shape2 = [None] * (max_len - len(inputs[1].shape)) + list(inputs[1].shape)
    for ax in range(-3, -max_len - 1, -1):
      if isinstance(shape1[ax], int) and shape1[ax] > 1:
        if shape2[ax] in [1, None]:
          var1.append(inputs[0].returnn_naming_entry.get_returnn_axis_description(ax))
        else:
          assert shape1[ax] == shape2[ax], f"dimensions are not broadcastable: {inputs[0].shape} vs. {inputs[1].shape}"
      if isinstance(shape2[ax], int) and shape2[ax] > 1:
        if shape1[ax] in [1, None]:
          var2.append(inputs[1].returnn_naming_entry.get_returnn_axis_description(ax))
        else:
          assert shape1[ax] == shape2[ax], f"dimensions are not broadcastable: {inputs[0].shape} vs. {inputs[1].shape}"

    return {"class": "dot", "red1": red1, "red2": red2, "var1": var1, "var2": var2, "from": sources}


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
