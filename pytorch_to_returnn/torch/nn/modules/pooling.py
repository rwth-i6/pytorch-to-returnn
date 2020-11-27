
from typing import Dict, Any, Optional
from returnn.tf.layers.basic import PoolLayer
from .module import Module
from .utils import _ntuple
from ..common_types import _size_any_t
from ...tensor import Tensor


class _MaxPoolNd(Module):
  nd: Optional[int] = None  # set in subclass
  return_indices: bool
  ceil_mode: bool

  def __init__(self, kernel_size: _size_any_t, stride: Optional[_size_any_t] = None,
               padding: _size_any_t = 0, dilation: _size_any_t = 1,
               return_indices: bool = False, ceil_mode: bool = False) -> None:
    super(_MaxPoolNd, self).__init__()
    self.kernel_size = _ntuple(self.nd)(kernel_size)
    self.stride = _ntuple(self.nd)(stride) if (stride is not None) else _ntuple(self.nd)(kernel_size)
    self.padding = _ntuple(self.nd)(padding)
    self.dilation = _ntuple(self.nd)(dilation)
    self.return_indices = return_indices
    self.ceil_mode = ceil_mode

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    assert not self.return_indices  # not implemented
    assert not self.ceil_mode  # not implemented (or maybe ignore?)
    assert all(p == 0 for p in self.padding)  # not implemented
    return {
      "class": "pool", "mode": "max", "from": self._get_input_layer_name(input),
      "pool_size": self.kernel_size,
      "dilation_rate": self.dilation,
      "strides": self.stride,
      "padding": "valid"}


class MaxPool1d(_MaxPoolNd):
  nd = 1


class MaxPool2d(_MaxPoolNd):
  nd = 2


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
