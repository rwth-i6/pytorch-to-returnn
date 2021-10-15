
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


class _Norm(Module):
  def __init__(self, axes: str, normalized_shape: _shape_t = None, eps: float = 1e-5, affine: bool = True) -> None:
    super(_Norm, self).__init__()
    self.axes = axes
    self.eps = eps
    self.affine = affine
    if self.affine:
      self.weight = Parameter(Tensor(normalized_shape))
      self.bias = Parameter(Tensor(normalized_shape))
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
    return {
      "class": "norm", "axes": self.axes, "epsilon": self.eps, "scale": self.affine, "bias": self.affine,
      "from": self._get_input_layer_name(input)}


class LayerNorm(_Norm):
  def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
    if isinstance(normalized_shape, numbers.Integral):
      normalized_shape = (normalized_shape,)
    self.normalized_shape = tuple(normalized_shape)
    self.elementwise_affine = elementwise_affine
    super(LayerNorm, self).__init__(axes="F", normalized_shape=normalized_shape, eps=eps, affine=elementwise_affine)


class GroupNorm(_Norm):
  def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True) -> None:
    assert num_groups == 1 or num_groups == num_channels, "Not implemented otherwise"
    self.num_groups = num_groups
    self.num_channels = num_channels

    axes = None
    if num_groups == 1:
      axes = "TF"
    elif num_groups == num_channels:
      axes = "T"
    super(GroupNorm, self).__init__(axes=axes, normalized_shape=num_channels, eps=eps, affine=affine)


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
