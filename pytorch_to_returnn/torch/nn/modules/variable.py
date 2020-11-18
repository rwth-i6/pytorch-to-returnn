
from ..parameter import Parameter
from .module import Module


class Variable(Module):
  def __init__(self, param: Parameter):
    super(Variable, self).__init__()
    assert isinstance(param, Parameter)
    self.param = param

  def create_returnn_layer_dict(self):
    return {"class": "variable", "add_batch_axis": False, "shape": self.param.shape}


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
