
from .module import Module


class BinaryOperator(Module):
  def __init__(self, kind: str):
    """
    :param kind: "add", "sub", "mul", "truediv"
    """
    super(BinaryOperator, self).__init__()
    self.kind = kind

  def create_returnn_layer_dict(self, *inputs):
    # TODO: implicitly merge dims...
    return {"class": "combine", "kind": self.kind, "from": inputs}


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
