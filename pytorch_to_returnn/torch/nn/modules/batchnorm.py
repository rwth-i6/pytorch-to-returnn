
from __future__ import annotations
import tensorflow as tf
import math
from typing import Optional, Dict, Any
from returnn.tf.layers.basic import BatchNormLayer
from .module import Module
from .utils import _single, _pair, _triple, _reverse_repeat_tuple, _ntuple
from ..common_types import _scalar_or_tuple_any_t, _size_1_t, _size_2_t, _size_3_t
from ..functional import zeros, ones, tensor
from ...tensor import Tensor
from ..parameter import Parameter
from .. import init


class _NormBase(Module):
  """Common base of _InstanceNorm and _BatchNorm"""
  _version = 2

  def __init__(
      self,
      num_features: int,
      eps: float = 1e-5,
      momentum: float = 0.1,
      affine: bool = True,
      track_running_stats: bool = True
  ) -> None:
    super(_NormBase, self).__init__()
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    self.affine = affine
    self.track_running_stats = track_running_stats
    if self.affine:
      self.weight = Parameter(Tensor(num_features))
      self.bias = Parameter(Tensor(num_features))
    else:
      self.register_parameter('weight', None)
      self.register_parameter('bias', None)
    if self.track_running_stats:
      self.register_buffer('running_mean', zeros(num_features))
      self.register_buffer('running_var', ones(num_features))
      self.register_buffer('num_batches_tracked', tensor(0, dtype="int64"))
    else:
      self.register_parameter('running_mean', None)
      self.register_parameter('running_var', None)
      self.register_parameter('num_batches_tracked', None)
    self.reset_parameters()

  def reset_running_stats(self) -> None:
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)
      self.num_batches_tracked.zero_()

  def reset_parameters(self) -> None:
    self.reset_running_stats()
    if self.affine:
      init.ones_(self.weight)
      init.zeros_(self.bias)

  def _check_input_dim(self, input):
    raise NotImplementedError

  def extra_repr(self):
    return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
           'track_running_stats={track_running_stats}'.format(**self.__dict__)

  def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs):
    version = local_metadata.get('version', None)

    if (version is None or version < 2) and self.track_running_stats:
      # at version 2: added num_batches_tracked buffer
      #               this should have a default value of 0
      num_batches_tracked_key = prefix + 'num_batches_tracked'
      if num_batches_tracked_key not in state_dict:
        state_dict[num_batches_tracked_key] = tensor(0, dtype="int64")

    super(_NormBase, self)._load_from_state_dict(
      state_dict, prefix, local_metadata, strict,
      missing_keys, unexpected_keys, error_msgs)


class _BatchNorm(_NormBase):
  def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
    super(_BatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    # Note: Momentum default is quite different. In RETURNN 0.99, in Torch 0.1.
    #  This is not because it's used the other way around.
    # Note: The other default behavior is different as well,
    #  i.e. we need to use `update_sample_only_in_training` and `delay_sample_update`.
    # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
    # https://github.com/pytorch/pytorch/blob/59605811488eb07b3b8bf70a5f0b4b56b34b4a61/aten/src/ATen/native/Normalization.cpp#L546
    return {
      "class": "batch_norm", "from": self._get_input_layer_name(input),
      "update_sample_only_in_training": True, "delay_sample_update": True,
      "param_version": 1,
      "momentum": self.momentum, "epsilon": self.eps}


class BatchNorm1d(_BatchNorm):
  def import_params_torch_to_returnn(self, *, layer: BatchNormLayer, torch_module: BatchNorm1d):
    import numpy
    session = tf.compat.v1.get_default_session()

    # The param names in our RETURNN layer are somewhat strange...
    def _get_param_by_name_postfix(name: str) -> tf.Variable:
      ps = [p for (name_, p) in layer.params.items() if name_.endswith(f"_{name}") or name_.endswith(f"/{name}")]
      assert len(ps) == 1, f"param name {name} not unique or found in layer {layer} with params {layer.params}"
      return ps[0]

    def _expand_dims(x: numpy.ndarray) -> numpy.ndarray:
      assert x.shape == (layer.output.dim,)
      out_shape = [
        layer.output.dim if i == layer.output.feature_dim_axis else 1
        for i in range(layer.output.batch_ndim)]
      return numpy.reshape(x, out_shape)

    def _convert(x: Tensor, name: str):
      tf_param = _get_param_by_name_postfix(name)
      values = _expand_dims(x.detach().cpu().numpy())
      tf_param.load(values, session=session)

    _convert(torch_module.running_mean, "mean")
    _convert(torch_module.running_var, "variance")
    _convert(torch_module.weight, "gamma")
    _convert(torch_module.bias, "beta")


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
