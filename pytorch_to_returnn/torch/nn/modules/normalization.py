
from __future__ import annotations
import math
import numbers
from typing import Dict, Any, Union, List
import tensorflow as tf
from returnn.tf.layers.basic import LayerNormLayer, NormLayer
from .module import Module
from ..parameter import Parameter
from ...tensor import Tensor, Size
from .. import init


_shape_t = Union[int, List[int], Size]


class LayerNorm(Module):
  def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
    super(LayerNorm, self).__init__()
    if isinstance(normalized_shape, numbers.Integral):
      normalized_shape = (normalized_shape,)
    self.normalized_shape = tuple(normalized_shape)
    self.eps = eps
    self.elementwise_affine = elementwise_affine
    if self.elementwise_affine:
      self.weight = Parameter(Tensor(*normalized_shape))
      self.bias = Parameter(Tensor(*normalized_shape))
    else:
      self.register_parameter('weight', None)
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    if self.elementwise_affine:
      init.ones_(self.weight)
      init.zeros_(self.bias)

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    assert len(self.normalized_shape) == 1  # not implemented otherwise
    axis = self._get_input_axis_to_returnn(input, axis=-1)
    assert axis == "F"  # not implemented otherwise. expect that feature-dim is last.
    assert self.elementwise_affine
    return {
      "class": "layer_norm", "from": self._get_input_layer_name(input),
      "epsilon": self.eps}

  def import_params_torch_to_returnn(self, *, layer: LayerNormLayer, torch_module: LayerNorm):
    assert isinstance(layer, LayerNormLayer)
    assert self.elementwise_affine
    assert len(self.normalized_shape) == 1  # not implemented otherwise
    session = tf.compat.v1.get_default_session()
    layer.params["scale"].load(torch_module.weight.detach().numpy(), session=session)
    layer.params["bias"].load(torch_module.bias.detach().numpy(), session=session)


class GroupNorm(Module):
  def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True) -> None:
    super(GroupNorm, self).__init__()
    assert num_groups == 1 or num_groups == num_channels, "Not implemented otherwise"
    self.num_groups = num_groups
    self.num_channels = num_channels
    self.eps = eps
    self.affine = affine
    if self.affine:
      self.weight = Parameter(Tensor(num_channels))
      self.bias = Parameter(Tensor(num_channels))
    else:
      self.register_parameter('weight', None)
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    if self.affine:
      init.ones_(self.weight)
      init.zeros_(self.bias)

  def import_params_torch_to_returnn(self, *, layer: NormLayer, torch_module: GroupNorm):
    assert isinstance(layer, NormLayer)
    if self.affine:
      session = tf.compat.v1.get_default_session()
      layer.params["scale"].load(torch_module.weight.detach().numpy(), session=session)
      layer.params["bias"].load(torch_module.bias.detach().numpy(), session=session)

  def create_returnn_layer_dict(self, input: Tensor):
    if self.num_groups == 1:
      axes = "TF"
    elif self.num_groups == self.num_channels:
      axes = "T"
    else:
      raise NotImplementedError(f"num groups {self.num_groups}, num channels {self.num_channels}")
    return {
      "class": "norm", "axes": axes, "epsilon": self.eps, "scale": self.affine, "bias": self.affine,
      "from": self._get_input_layer_name(input)}


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
