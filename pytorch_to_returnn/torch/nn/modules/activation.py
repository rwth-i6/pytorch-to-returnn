
from .module import Module
from .. import functional as F
from ...tensor import Tensor


class Tanh(Module):
  def forward(self, input: Tensor) -> Tensor:
    return F.tanh(input)


class LeakyReLU(Module):
  def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
    super(LeakyReLU, self).__init__()
    self.negative_slope = negative_slope
    self.inplace = inplace

  def forward(self, input: Tensor) -> Tensor:
    return F.leaky_relu(input, self.negative_slope, self.inplace)


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
