
from typing import List
from .module import Module
from ...tensor import Tensor


class Norm(Module):
  is_original_torch_module = False

  def __init__(self, *, axes: List[int], p: float = 2, keepdims: bool = False):
    super(Norm, self).__init__()
    self.axes = axes
    self.p = p
    self.keepdims = keepdims

  def create_returnn_layer_dict(self, input: Tensor):
    return {
      "class": "math_norm", "p": self.p, "keep_dims": self.keepdims,
      "axes": [self._get_input_axis_to_returnn(input, axis) for axis in self.axes],
      "from": self._get_input_layer_name(input)}


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
