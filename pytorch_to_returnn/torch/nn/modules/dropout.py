
from typing import Dict, Any
from .module import Module
from ...tensor import Tensor


class _DropoutNd(Module):
  p: float
  inplace: bool

  def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
    super(_DropoutNd, self).__init__()
    if p < 0 or p > 1:
      raise ValueError("dropout probability has to be between 0 and 1, "
                       "but got {}".format(p))
    self.p = p
    self.inplace = inplace  # just ignore...


class Dropout(_DropoutNd):
  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    return {"class": "dropout", "dropout": self.p, "from": self._get_input_layer_name(input)}


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
