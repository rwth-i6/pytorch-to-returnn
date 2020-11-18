
from typing import Optional, Tuple, Any
from .module import Module
from ...tensor import Tensor


class BinaryOperator(Module):
  def __init__(self, kind: str):
    """
    :param kind: "add", "sub", "mul", "truediv"
    """
    super(BinaryOperator, self).__init__()
    self.kind = kind

  def create_returnn_layer_dict(self, *inputs: Tensor):
    # TODO: implicitly merge dims...
    return {
      "class": "combine", "kind": self.kind,
      "from": [self._get_input_layer_name(input) for input in inputs]}


class Reciprocal(Module):
  """
  1/x or 1/max(eps,x)
  """
  def __init__(self, eps: Optional[float] = None):
    super(Reciprocal, self).__init__()
    self.eps = eps

  def create_returnn_layer_dict(self, input: Tensor):
    x = "source(0)"
    if self.eps is not None:
      x = f"maximum_with_identity_grad({x})"
    return {
      "class": "eval", "eval": f"tf_compat.v1.reciprocal({x})",
      "from": self._get_input_layer_name(input)}


class Max(Module):
  pass  # TODO


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
