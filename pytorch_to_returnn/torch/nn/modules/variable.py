
from returnn.tf.layers.base import LayerBase
from typing import Tuple, List
from ...tensor import Tensor
from ..parameter import Parameter
from .module import Module


class Variable(Module):
  is_original_torch_module = False

  def __init__(self, param: Parameter):
    super(Variable, self).__init__()
    assert isinstance(param, Parameter)
    self.param = param

  def create_returnn_layer_dict(self, *inputs):  # ignore inputs
    return {"class": "variable", "add_batch_axis": False, "shape": self.param.shape}

  def make_output_tensor_from_returnn(self, inputs_flat: List[Tensor], layer: LayerBase) -> Tensor:
    return self.param


class Constant(Module):
  is_original_torch_module = False

  def __init__(self, value: Tensor):
    super(Constant, self).__init__()
    assert isinstance(value, Tensor)
    self.value = value

  def create_returnn_layer_dict(self, *inputs):  # ignore inputs
    return {"class": "constant", "value": self.value.numpy()}

  def make_output_tensor_from_returnn(self, inputs_flat: List[Tensor], layer: LayerBase) -> Tensor:
    return self.value


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
