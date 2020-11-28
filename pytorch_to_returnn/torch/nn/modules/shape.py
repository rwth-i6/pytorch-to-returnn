
"""
This file is originally called "flatten.py", but we extend it by further reshape modules.

Note on PyTorch vs RETURNN/TF terminology:

*Axis* in TF == *dim* in PyTorch.
*Dim(s)* in TF == *size* in PyTorch.
"""

from typing import Tuple, Union, Collection, Dict, Any
from .module import Module
from ...tensor import Tensor, Size


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
  def __init__(self, dim: Union[int, str], unflattened_size: Union[Size, _NamedShape]) -> None:
    super(Unflatten, self).__init__()
    self.dim = dim
    self.unflattened_size = unflattened_size

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    assert isinstance(self.dim, int)  # not implemented otherwise
    return {
      "class": "split_dims", "from": self._get_input_layer_name(input),
      "axis": self._get_input_axis_to_returnn(input, self.dim),
      "dims": self.unflattened_size}


class SplitDims(Unflatten):
  """
  Alias, RETURNN consistent name
  """
  is_original_torch_module = False


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
