
from __future__ import annotations
import math
from typing import Dict, Any, Tuple, List
import tensorflow as tf
from returnn.tf.layers.basic import LayerBase, LinearLayer
from .module import Module
from ..parameter import Parameter
from ...tensor import Tensor
from .. import init
from ....naming import Naming, TensorEntry


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



class Matmul(Module):
  """
  Maps to RETURNN DotLayer
  """
  is_original_torch_module = False

  def _get_output_shape_from_returnn(self, inputs_flat: List[Tensor], layer: LayerBase
                                     ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    assert len(inputs_flat) == 2
    input1, input2 = inputs_flat
    assert all(isinstance(input_, Tensor) for input_ in [input1, input2])
    assert all(len(input_.shape) >= 2 for input_ in [input1, input2]), "not implemented otherwise"

    max_len = max(len(input1.shape), len(input2.shape))
    shape1 = [None] * (max_len - len(input1.shape)) + list(input1.shape)
    shape2 = [None] * (max_len - len(input2.shape)) + list(input2.shape)
    shape = []
    for ax in range(max_len - 2):
      if shape1[ax] == shape2[ax]:
        shape.append(shape1[ax])
      elif isinstance(shape1[ax], int) and shape1[ax] > 1:
        assert shape2[ax] in [1, None], f"dimensions are not broadcastable: {input1.shape} vs. {input2.shape}"
        shape.append(shape1[ax])
      elif isinstance(shape2[ax], int) and shape2[ax] > 1:
        assert shape1[ax] in [1, None], f"dimensions are not broadcastable: {input1.shape} vs. {input2.shape}"
        shape.append(shape2[ax])

    shape += [shape1[-2], shape2[-1]]
    shape = tuple(shape)
    returnn_axis_from_torch_axis = {i: i for i in range(len(shape))}
    return shape, returnn_axis_from_torch_axis

  def create_returnn_layer_dict(self, *inputs: Tensor, **kwargs) -> Dict[str, Any]:
    sources = [self._get_input_layer_name(source) for source in inputs]
    assert len(sources) == 2

    assert len(inputs[0].shape) >= 2, "not implemented otherwise"
    assert len(inputs[1].shape) >= 2, "not implemented otherwise"
    red1 = self._get_input_axis_to_returnn(inputs[0], -1)
    red2 = self._get_input_axis_to_returnn(inputs[1], -2)
    var1 = [self._get_input_axis_to_returnn(inputs[0], -2)]
    var2 = [self._get_input_axis_to_returnn(inputs[1], -1)]

    max_len = max(len(inputs[0].shape), len(inputs[1].shape))
    shape1 = [None] * (max_len - len(inputs[0].shape)) + list(inputs[0].shape)
    shape2 = [None] * (max_len - len(inputs[1].shape)) + list(inputs[1].shape)
    for ax in range(-3, -max_len - 1, -1):
      if isinstance(shape1[ax], int) and shape1[ax] > 1:
        if shape2[ax] in [1, None]:
          var1.insert(0, self._get_input_axis_to_returnn(inputs[0], ax))
        else:
          assert shape1[ax] == shape2[ax], f"dimensions are not broadcastable: {inputs[0].shape} vs. {inputs[1].shape}"
      if isinstance(shape2[ax], int) and shape2[ax] > 1:
        if shape1[ax] in [1, None]:
          var2.insert(0, self._get_input_axis_to_returnn(inputs[1], ax))
        else:
          assert shape1[ax] == shape2[ax], f"dimensions are not broadcastable: {inputs[0].shape} vs. {inputs[1].shape}"

    return {"class": "dot", "red1": red1, "red2": red2, "var1": var1, "var2": var2, "from": sources}


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
