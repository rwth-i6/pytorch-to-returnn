
from ...tensor import Tensor
from .module import Module
from typing import Union, Tuple
from .utils import _pair, _quadruple, _ntuple
from .. import functional as F
from ..common_types import _size_2_t, _size_4_t, _size_6_t


class _PadReturnn(Module):
  padding = None  # set by subclasses
  mode = None  # set by subclasses
  value = None  # set by subclasses

  def forward(self, input: Tensor) -> Tensor:
    pass  # TODO


class _ConstantPadNd(Module):
  padding = None  # set by subclasses
  value = None  # set by subclasses

  def __init__(self, value: float) -> None:
    super(_ConstantPadNd, self).__init__()
    self.value = value

  def forward(self, input: Tensor) -> Tensor:
    return F.pad(input, self.padding, 'constant', self.value)


class ConstantPad1d(_ConstantPadNd):
  def __init__(self, padding: _size_2_t, value: float):
    super(ConstantPad1d, self).__init__(value)
    self.padding = _pair(padding)


class _ReflectionPadNd(Module):
  padding = None  # set by subclasses

  def forward(self, input: Tensor) -> Tensor:
    return F.pad(input, self.padding, 'reflect')


class ReflectionPad1d(_ReflectionPadNd):
  def __init__(self, padding: _size_2_t):
    super(ReflectionPad1d, self).__init__()
    self.padding = _pair(padding)


class _ReplicationPadNd(Module):
  padding = None  # set by subclasses

  def forward(self, input: Tensor) -> Tensor:
    return F.pad(input, self.padding, 'replicate')


class ReplicationPad1d(_ReplicationPadNd):
  def __init__(self, padding: _size_2_t) -> None:
    super(ReplicationPad1d, self).__init__()
    self.padding = _pair(padding)


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
