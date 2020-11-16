
from typing import Dict, Any
from .module import Module
from ...tensor import Tensor


class _ActivationReturnn(Module):
  func_name: str

  def create_returnn_layer_dict(self, input: str) -> Dict[str, Any]:
    return {"class": "activation", "activation": self.func_name, "from": input}


class Tanh(_ActivationReturnn):
  func_name = "tanh"


class LeakyReLU(Module):
  def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
    super(LeakyReLU, self).__init__()
    self.negative_slope = negative_slope
    assert not inplace  # not supported/implemented -- see :doc:`Unsupported`

  def create_returnn_layer_dict(self, input: str) -> Dict[str, Any]:
    return {"class": "eval", "eval": f"tf.nn.leaky_relu(source(0), alpha={self.negative_slope})", "from": input}


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
