
"""
This file is originally called "flatten.py", but we extend it by further reshape modules.

Note on PyTorch vs RETURNN/TF terminology:

*Axis* in TF == *dim* in PyTorch.
*Dim(s)* in TF == *size* in PyTorch.
"""

from __future__ import annotations
from typing import Tuple, List, Union, Collection, Dict, Any
from returnn.tf.layers.basic import LayerBase
from .module import Module
from ...tensor import Tensor, Size
from ....naming import Naming


_NamedShape = Tuple[Tuple[str, int]]


class MergeDims(Module):
  is_original_torch_module = False

  def __init__(self, dims: Collection[int]):
    super(MergeDims, self).__init__()
    self.dims = dims

  @classmethod
  def generic_create_returnn_layer_dict(cls, input: Tensor, dims: Collection[int]) -> Dict[str, Any]:
    assert isinstance(dims, (tuple, list))
    return {
      "class": "merge_dims", "from": cls._get_input_layer_name(input),
      "axes": [cls._get_input_axis_to_returnn(input, axis=axis) for axis in dims],
      "keep_order": True}

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    return self.generic_create_returnn_layer_dict(input=input, dims=self.dims)


class Squeeze(Module):
  is_original_torch_module = False

  def __init__(self, dim: int):
    super(Squeeze, self).__init__()
    self.dim = dim

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    return {
      "class": "squeeze", "from": self._get_input_layer_name(input),
      "axis": self._get_input_axis_to_returnn(input, self.dim)}


class Flatten(Module):
  """
  Originally in flatten.py.
  """
  def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
    super(Flatten, self).__init__()
    self.start_dim = start_dim
    self.end_dim = end_dim

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    def _dim(d: int) -> int:
      assert -input.ndim <= d < input.ndim
      if d < 0:
        d += input.ndim
      assert 0 <= d < input.ndim
      return d
    start_dim = _dim(self.start_dim)
    end_dim = _dim(self.end_dim)
    assert start_dim <= end_dim
    dims = list(range(start_dim, end_dim + 1))  # end_dim is inclusive
    return MergeDims.generic_create_returnn_layer_dict(input=input, dims=dims)


class Unflatten(Module):
  """
  Originally in flatten.py.
  """
  def __init__(self,
               dim: Union[int, str],
               unflattened_size: Union[Size, _NamedShape, Tuple[int, ...], List[int]]) -> None:
    super(Unflatten, self).__init__()
    self.dim = dim
    self.unflattened_size = unflattened_size

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    assert isinstance(self.dim, int)  # not implemented otherwise
    dims = list(self.unflattened_size)
    # Introduce -1 again to dims, such that we can handle dynamic axes in RETURNN.
    non_one_dims = [d for d in dims if d != 1]
    if len(non_one_dims) == 1:
      dims[dims.index(non_one_dims[0])] = -1
    return {
      "class": "split_dims", "from": self._get_input_layer_name(input),
      "axis": self._get_input_axis_to_returnn(input, self.dim),
      "dims": dims}

  def _get_output_shape_from_returnn(self,
                                     inputs_flat: List[Tensor], layer: LayerBase
                                     ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    """
    We can exactly infer this, so we don't need to rely on the default heuristics.

    :return: (torch_shape, returnn_axis_from_torch_axis).
    """
    naming = Naming.get_instance()
    assert len(inputs_flat) == 1
    x = inputs_flat[0]
    x_entry = naming.register_tensor(x)
    dim = self.dim
    assert -len(x.shape) <= dim < len(x.shape)
    if dim < 0:
      dim += len(x.shape)
    assert 0 <= dim < len(x.shape)
    unflatten_size = list(self.unflattened_size)
    if any(d == -1 for d in unflatten_size):
      rem_dim = x.shape[dim]
      for d in unflatten_size:
        if d == -1:
          continue
        assert rem_dim % d == 0
        rem_dim //= d
      unflatten_size[unflatten_size.index(-1)] = rem_dim
    assert all(d > 0 for d in unflatten_size)
    out_torch_shape = x.shape[:dim] + tuple(unflatten_size) + x.shape[dim + 1:]

    old_returnn_split_axis = x_entry.returnn_axis_from_torch_axis[dim]
    out_returnn_axis_from_torch_axis = {}
    for i in range(len(out_torch_shape)):
      unflatten_idx = None
      if i < dim:
        old_torch_axis = i
      elif i > dim + len(unflatten_size):
        old_torch_axis = i - len(unflatten_size) + 1
      else:  # i >= dim and ..., within the unflattened dims
        unflatten_idx = i - dim
        old_torch_axis = dim
      old_returnn_axis = x_entry.returnn_axis_from_torch_axis[old_torch_axis]
      if old_returnn_axis < old_returnn_split_axis:
        returnn_axis = old_returnn_axis
      elif old_returnn_axis > old_returnn_split_axis:
        returnn_axis = old_returnn_axis + len(unflatten_size) - 1
      else:
        returnn_axis = old_returnn_axis + unflatten_idx
      out_returnn_axis_from_torch_axis[i] = returnn_axis
    return out_torch_shape, out_returnn_axis_from_torch_axis


class SplitDims(Unflatten):
  """
  Alias, RETURNN consistent name
  """
  is_original_torch_module = False


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
