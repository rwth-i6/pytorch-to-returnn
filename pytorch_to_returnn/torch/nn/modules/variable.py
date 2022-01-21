
from __future__ import annotations
import numpy
import tensorflow as tf
from returnn.tf.layers.basic import LayerBase, VariableLayer
from typing import Tuple, List, Dict, Optional
from ...tensor import Tensor
from ..parameter import Parameter
from .module import Module
from ....naming import Naming


class Variable(Module):
  is_original_torch_module = False

  def __init__(self, param: Parameter, parent_mod: Optional[Tuple[Module, str]] = None):
    super(Variable, self).__init__()
    assert isinstance(param, Parameter)
    self.param = param
    self.parent_mod = parent_mod

  def create_returnn_layer_dict(self, *inputs):  # ignore inputs
    return {"class": "variable", "add_batch_axis": False, "shape": self.param.shape}

  def make_output_tensor_from_returnn(self, inputs_flat: List[Tensor], layer: LayerBase) -> Tensor:
    naming = Naming.get_instance()
    values = None
    if naming.import_params_from_torch_namespace and self.parent_mod:
      if not layer.get_absolute_name().startswith("."):  # temp layer
        parent_mod, param_name = self.parent_mod
        mod_abs_name = naming.get_module_abs_id_name(parent_mod)
        torch_mod = naming.import_params_from_torch_namespace.get_module_by_abs_id_name(mod_abs_name)
        torch_param = getattr(torch_mod, param_name)
        values = torch_param.detach().numpy()
    if values is None:
      values = self.param.detach().numpy()
    assert isinstance(layer, VariableLayer)
    assert len(layer.params) == 1
    tf_param = list(layer.params.values())[0]
    session = tf.compat.v1.get_default_session()
    tf_param.load(values, session=session)
    return self.param


class Constant(Module):
  is_original_torch_module = False

  def __init__(self, value: Tensor):
    super(Constant, self).__init__()
    assert isinstance(value, Tensor)
    self.value = value

  def _get_batch_dim(self) -> Optional[int]:
    tensor = self.value
    batch_dims = [i for i, d in enumerate(tensor.shape) if d.is_batch_dim]
    assert len(batch_dims) <= 1  # not implemented otherwise
    if batch_dims:
      return batch_dims[0]
    return None

  def create_returnn_layer_dict(self, *inputs):  # ignore inputs
    tensor = self.value
    value = tensor.numpy()
    batch_axis = self._get_batch_dim()
    if batch_axis is not None:
      # We remove the batch axis now, and set with_batch_dim below.
      # But we need to check whether this is valid.
      value = numpy.moveaxis(value, batch_axis, 0)
      for i in range(1, value.shape[0]):
        numpy.testing.assert_equal(value[0], value[i])
      value = numpy.array(value[0])  # remove batch axis
    assert isinstance(value, numpy.ndarray)
    # Simplify representation in these simple cases.
    if not value.shape:  # scalar
      if value.dtype == "int32":
        value = int(value)
      elif value.dtype == "float32":
        value = float(value)
    d = {"class": "constant", "value": value}
    if batch_axis is not None:
      d["with_batch_dim"] = True
      naming = Naming.get_instance()
      call = naming.module_call_stack[-1]
      assert call.module.module is self
      # Add some of the network inputs as the dependency, to get the batch-dim.
      call.inputs_flat = [naming.inputs[0]]
      # Note: I wonder whether we maybe need others here...?
    return d

  def make_output_tensor_from_returnn(self, inputs_flat: List[Tensor], layer: LayerBase) -> Tensor:
    naming = Naming.get_instance()
    tensor = self.value
    entry = naming.register_tensor(tensor)
    entry.returnn_data = layer.output
    batch_axis = self._get_batch_dim()
    if batch_axis is not None:
      # Batch axis will be moved to top.
      def _new_axis(old_axis: int) -> int:
        if old_axis == batch_axis:
          return 0
        if old_axis < batch_axis:
          return old_axis + 1
        return old_axis
      entry.returnn_axis_from_torch_axis = {i: _new_axis(i) for i in range(tensor.ndim)}
    else:
      entry.returnn_axis_from_torch_axis = {i: i for i in range(tensor.ndim)}
    return tensor


class FullStatic(Module):
  is_original_torch_module = False

  def __init__(self, fill_value, dtype):
    super(FullStatic, self).__init__()
    self.fill_value = fill_value
    self.dtype = dtype

  def create_returnn_layer_dict(self, size):
    # We require the size to contain some static information.
    assert isinstance(size, (tuple, list))  # not implemented otherwise
    naming = Naming.get_instance()

    def _convert_dim(x):
      if isinstance(x, int):
        return x
      if isinstance(x, Tensor):
        tensor_entry = naming.tensors[x]
        assert x.is_defined and tensor_entry.is_const and tensor_entry.is_dim
        return tensor_entry.is_dim
      raise TypeError(f"FullStatic: cannot handle dim {x!r} of type {type(x)}")

    return {
      "class": "constant", "shape": [_convert_dim(x) for x in size],
      "value": self.fill_value, "dtype": self.dtype}

  def make_output_tensor_from_returnn(self, inputs_flat: List[Tensor], layer: LayerBase) -> Tensor:
    from ..._C import SizeValue
    naming = Naming.get_instance()
    size = inputs_flat

    def _convert_dim(x):
      if isinstance(x, SizeValue):
        raise Exception(f"SizeValue {x} not expected, should be a Tensor, via Naming._make_tensor")
      if isinstance(x, int):
        return x
      if isinstance(x, Tensor):
        assert x.is_defined and x.shape == () and x.dtype.name.startswith("int")
        return int(x.numpy())
      raise TypeError(f"invalid dim {x!r} type {type(x)}")

    size = [_convert_dim(x) for x in size]
    from ..._C import from_numpy
    tensor = from_numpy(numpy.full(size, self.fill_value, dtype=self.dtype))
    entry = naming.register_tensor(tensor)
    entry.is_const = True
    entry.returnn_data = layer.output
    entry.returnn_axis_from_torch_axis = {i: i for i in range(tensor.ndim)}
    return tensor


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
