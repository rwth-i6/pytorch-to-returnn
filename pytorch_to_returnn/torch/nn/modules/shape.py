
"""
This file is originally called "flatten.py", but we extend it by further reshape modules.

Note on PyTorch vs RETURNN/TF terminology:

*Axis* in TF == *dim* in PyTorch.
*Dim(s)* in TF == *size* in PyTorch.
"""

from __future__ import annotations
from typing import Tuple, List, Union, Collection, Dict, Any, Optional
from functools import reduce
import operator
from returnn.tf.layers.basic import LayerBase, MergeDimsLayer, FlattenBatchLayer
from .module import Module
from ...tensor import Tensor, Size
from ....naming import Naming


_NamedShape = Tuple[Tuple[str, int]]


class MergeDims(Module):
  """
  Maps to RETURNN MergeDimsLayer, if batch axis is not involved.
  In case batch axis is involved, this uses RETURNN FlattenBatchLayer.

  Used by Flatten, and thus indirectly by various kinds of reshape.
  """
  is_original_torch_module = False

  def __init__(self, dims: Collection[int]):
    super(MergeDims, self).__init__()
    self.dims = dims

  @classmethod
  def generic_create_returnn_layer_dict(cls, input: Tensor, dims: Collection[int]) -> Dict[str, Any]:
    naming = Naming.get_instance()
    in_ = naming.register_tensor(input)
    assert in_.returnn_data
    assert isinstance(dims, (tuple, list))

    def _pos_dim(d: int) -> int:
      assert -len(input.shape) <= d < len(input.shape)
      if d < 0:
        d += len(input.shape)
      assert 0 <= d < len(input.shape)
      return d

    dims = [_pos_dim(d) for d in dims]
    returnn_axes = [in_.returnn_axis_from_torch_axis[d] for d in dims]
    if in_.returnn_data.have_batch_axis() and in_.returnn_data.batch_dim_axis in returnn_axes:
      if set(in_.returnn_data.get_dynamic_axes()).intersection(returnn_axes):
        # We merge batch with some other dynamic axis.
        batch_major = returnn_axes[0] == in_.returnn_data.batch_dim_axis
        returnn_axes.pop(in_.returnn_data.batch_dim_axis)
        assert len(returnn_axes) == 1  # not implemented otherwise
        return {
          "class": "flatten_batch",
          "from": cls._get_input_layer_name(input),
          "axis": cls._get_input_axis_to_returnn(input, axis=in_.torch_axis_from_returnn_axis[returnn_axes[0]]),
          "batch_major": batch_major}

    return {
      "class": "merge_dims", "from": cls._get_input_layer_name(input),
      "axes": [cls._get_input_axis_to_returnn(input, axis=axis) for axis in dims],
      "keep_order": True}

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    return self.generic_create_returnn_layer_dict(input=input, dims=self.dims)

  @classmethod
  def generic_get_output_shape_from_returnn(cls,
                                            input: Tensor,
                                            layer: Union[MergeDimsLayer, FlattenBatchLayer],
                                            dims: Collection[int]
                                            ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    """
    We can exactly infer this, so we don't need to rely on the default heuristics.
    Also, we must do this, in case of dynamic axes.

    :return: (torch_shape, returnn_axis_from_torch_axis).
    """
    naming = Naming.get_instance()
    in_ = naming.register_tensor(input)
    assert in_.returnn_data
    assert isinstance(dims, (tuple, list))

    def _pos_dim(d: int) -> int:
      assert -len(input.shape) <= d < len(input.shape)
      if d < 0:
        d += len(input.shape)
      assert 0 <= d < len(input.shape)
      return d

    # Switch to TF terminology. axis, not dim.
    in_torch_axes = [_pos_dim(d) for d in dims]
    in_returnn_axes = [in_.returnn_axis_from_torch_axis[i] for i in in_torch_axes]
    out_returnn_reduced_axis = min(in_returnn_axes)
    out_size = reduce(operator.mul, [input.shape[i] for i in in_torch_axes], 1)
    out_torch_reduced_axis = min(in_torch_axes)
    out_torch_shape = [input.shape[i] for i in range(input.ndim) if i not in in_torch_axes]
    out_torch_shape = tuple(
      out_torch_shape[:out_torch_reduced_axis] + [out_size] + out_torch_shape[out_torch_reduced_axis:])

    out_returnn_axis_from_torch_axis = {out_torch_reduced_axis: out_returnn_reduced_axis}
    for in_torch_axis in range(len(input.shape)):
      if in_torch_axis in in_torch_axes:
        continue  # reduced. covered already
      elif in_torch_axis < min(in_torch_axes):
        torch_offset = 0
      elif in_torch_axis > max(in_torch_axes):
        torch_offset = len(in_torch_axes) - 1
      else:
        torch_offset = len([a for a in in_torch_axes if a < in_torch_axis])
        assert 0 < torch_offset < len(in_torch_axes), f"in_torch_axis={in_torch_axis}?"  # min/max already covered
        torch_offset -= 1
      out_torch_axis = in_torch_axis - torch_offset
      assert out_torch_axis not in out_returnn_axis_from_torch_axis

      in_returnn_axis = in_.returnn_axis_from_torch_axis[in_torch_axis]
      assert in_returnn_axis not in in_returnn_axes  # not reduced axis
      if in_returnn_axis < min(in_returnn_axes):
        returnn_offset = 0
      elif in_returnn_axis > max(in_returnn_axes):
        returnn_offset = len(in_returnn_axes) - 1
      else:
        returnn_offset = len([a for a in in_returnn_axes if a < in_returnn_axis])
        assert 0 < returnn_offset < len(in_returnn_axes), f"in_returnn_axis={in_returnn_axis}?"
        returnn_offset -= 1
      out_returnn_axis = in_returnn_axis - returnn_offset
      out_returnn_axis_from_torch_axis[out_torch_axis] = out_returnn_axis

    if isinstance(layer, MergeDimsLayer):
      pass
    elif isinstance(layer, FlattenBatchLayer):
      # The reduced axis was moved to front, i.e. out_returnn_reduced_axis = 0.
      if out_returnn_reduced_axis != 0:  # only needs change if it was not 0 already
        # Basically movedim(out, out_returnn_axis_from_torch_axis, 0).
        def _new_axis(old: int) -> int:
          if old == out_returnn_reduced_axis:
            return 0
          if old < out_returnn_reduced_axis:
            return old + 1
          return old
        out_returnn_axis_from_torch_axis = {i: _new_axis(j) for (i, j) in out_returnn_axis_from_torch_axis.items()}
    else:
      raise TypeError(f"Unexpected layer {layer}, with input {input} and dims {dims}")

    return out_torch_shape, out_returnn_axis_from_torch_axis

  def _get_output_shape_from_returnn(self,
                                     inputs_flat: List[Tensor], layer: Union[MergeDimsLayer, FlattenBatchLayer]
                                     ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    assert len(inputs_flat) == 1
    return self.generic_get_output_shape_from_returnn(input=inputs_flat[0], layer=layer, dims=self.dims)


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

  def _get_dims(self, input: Tensor):
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
    return dims

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    dims = self._get_dims(input)
    return MergeDims.generic_create_returnn_layer_dict(input=input, dims=dims)

  def _get_output_shape_from_returnn(self,
                                     inputs_flat: List[Tensor], layer: Union[MergeDimsLayer, FlattenBatchLayer]
                                     ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    assert len(inputs_flat) == 1
    input = inputs_flat[0]
    dims = self._get_dims(input)
    return MergeDims.generic_get_output_shape_from_returnn(input=input, layer=layer, dims=dims)


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


class Split(Module):
  """
  tf.split, SplitLayer in RETURNN
  """
  is_original_torch_module = False

  def __init__(self, *, dim: int, num_splits: Optional[int] = None, size_splits: Optional[List[int]] = None):
    super(Split, self).__init__()
    self.dim = dim
    assert num_splits or size_splits
    self.num_splits = num_splits
    self.size_splits = size_splits

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    d = {
      "class": "split", "from": self._get_input_layer_name(input),
      "axis": self._get_input_axis_to_returnn(input, axis=self.dim)}
    if self.num_splits is not None:
      d["num_splits"] = self.num_splits
    if self.size_splits is not None:
      d["size_splits"] = self.size_splits
    return d

  def make_structured_returnn_output(self, output: Tensor) -> List[Tensor]:
    from .operator import GetSublayer
    if self.num_splits is not None:
      num_splits = self.num_splits
    else:
      num_splits = len(self.size_splits)
    return [GetSublayer(sub_layer=f"{i}")(output) for i in range(num_splits)]


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
