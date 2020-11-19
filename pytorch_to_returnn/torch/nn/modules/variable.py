
from returnn.tf.layers.base import LayerBase
from typing import Tuple
from ...tensor import Tensor
from ..parameter import Parameter
from .module import Module


class Variable(Module):
  def __init__(self, param: Parameter):
    super(Variable, self).__init__()
    assert isinstance(param, Parameter)
    self.param = param

  def create_returnn_layer_dict(self, *inputs):  # ignore inputs
    return {"class": "variable", "add_batch_axis": False, "shape": self.param.shape}

  def _make_output_tensor_from_returnn(self, inputs: Tuple[Tensor, ...], layer: LayerBase) -> Tensor:
    return self.param


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
