
from ...tensor import Tensor
from .module import Module
from typing import Union, Tuple, Optional
from .utils import _pair, _quadruple, _ntuple
from ..common_types import _size_2_t, _size_4_t, _size_6_t


class GenericPadNd(Module):
  padding: Tuple[int, ...] = None  # set by instance
  mode: str = None  # set by subclass
  value: float = None  # set by subclass instance
  nd: int = None  # set by subclass

  def __init__(self, *, padding: Tuple[int, ...], mode: Optional[str] = None, value: Optional[float] = None):
    super(GenericPadNd, self).__init__()
    self.padding = padding
    if mode is not None:
      self.mode = mode
    if value is not None:
      self.value = value

  def create_returnn_layer_dict(self, input: Tensor):
    assert self.mode
    assert self.mode != "replicate"  # not implemented
    # PyTorch assumes the input to be in batch-feature-major.
    # E.g. for 1D, it assumes input (N, C, W_in),
    # and produces output (N, C, W_out) with W_out = W_in + padding_left + padding_right.
    # For 2D, it assumes input (N, C, H_in, W_in).
    # For 3D, it assumes input (N, C, D_in, H_in, W_in).
    # I.e. does padding in the spatial axes.
    d = {
      "class": "pad", "mode": self.mode, "axes": "spatial", "padding": self.padding,
      "from": self._get_input_layer_name(input)}
    if self.mode == "constant":
      d["value"] = self.value
    return d


class _ConstantPadNd(GenericPadNd):
  mode = "constant"

  def __init__(self, *, padding: Tuple[int, ...], value: float) -> None:
    super(_ConstantPadNd, self).__init__(padding=padding)
    self.value = value


class ConstantPad1d(_ConstantPadNd):
  nd = 1

  def __init__(self, padding: _size_2_t, value: float):
    super(ConstantPad1d, self).__init__(padding=_pair(padding), value=value)


class _ReflectionPadNd(GenericPadNd):
  mode = "reflect"


class ReflectionPad1d(_ReflectionPadNd):
  nd = 1

  def __init__(self, padding: _size_2_t):
    super(ReflectionPad1d, self).__init__(padding=_pair(padding))


class _ReplicationPadNd(GenericPadNd):
  mode = "replicate"


class ReplicationPad1d(_ReplicationPadNd):
  nd = 1

  def __init__(self, padding: _size_2_t) -> None:
    super(ReplicationPad1d, self).__init__(padding=_pair(padding))


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
