
from __future__ import annotations
import math
from typing import Dict, Any
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


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
