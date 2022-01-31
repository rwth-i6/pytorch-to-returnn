
"""
Dummies...
"""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING, Union, Any, Tuple, List, Optional
from returnn.tf.util.data import Dim
from ..naming import Naming
if TYPE_CHECKING:
  from .tensor import Tensor


def device(name):
  return name


# noinspection PyPep8Naming
class dtype:
  def __init__(self, arg: Union[str, dtype]):
    if isinstance(arg, str):
      self.name = arg
    elif isinstance(arg, dtype):
      self.name = arg.name  # type: str
    else:
      raise TypeError(f"unexpected arg {arg}")

  def __str__(self):
    return f"torch.{self.name}"

  def __eq__(self, other: Union[str, dtype, Any]):
    if isinstance(other, str):
      return self.canonical_name == dtype(other).canonical_name
    if isinstance(other, dtype):
      return self.canonical_name == other.canonical_name
    return False

  def __ne__(self, other: Union[str, dtype, Any]):
    return not (self == other)

  def __hash__(self):
    return hash(self.name)

  @property
  def is_signed(self) -> bool:
    if self.name == "bool":
      return False
    return self.name[:1] != "u"

  @property
  def is_complex(self) -> bool:
    if self.name == "cfloat":
      return True
    if self.name == "cdouble":
      return True
    return self.name.startswith("complex")

  @property
  def is_floating_point(self) -> bool:
    if self.is_complex:
      return True
    if self.name == "double":
      return True
    if self.name == "half":
      return True
    if self.name.startswith("float"):
      return True
    if self.name.startswith("bfloat"):
      return True
    return False

  @property
  def category(self) -> str:
    # https://pytorch.org/docs/stable/tensor_attributes.html
    # complex > floating > integral > boolean
    if self.is_complex:
      return "complex"
    if self.is_floating_point:
      return "float"
    if self.name == "bool":
      return "bool"
    return "int"

  @property
  def category_int(self) -> int:
    return {"bool": 0, "int": 1, "float": 2, "complex": 3}[self.category]

  @property
  def canonical_name(self) -> str:
    """
    Can be used for __eq__.
    """
    d = {
      "half": "float16",
      "float": "float32",
      "double": "float64",
      "cfloat": "complex64",
      "ddouble": "complex128",
      "short": "int16",
      "int": "int32",
      "long": "int64",
    }
    return d.get(self.name, self.name)

  @property
  def bit_size(self) -> int:
    if self.name == "bool":
      return 8
    name = self.canonical_name
    if "128" in name:
      return 128
    if "64" in name:
      return 64
    if "32" in name:
      return 32
    if "16" in name:
      return 16
    if "8" in name:
      return 8
    raise TypeError(f"unexpected dtype {self.name}")


class Size(tuple):  # type: Tuple[SizeValue, ...]
  pass


class SizeValue(int):
  """
  We extend this, to store extra information, e.g. corresponding RETURNN dim tags.
  """
  def __new__(cls, x: int, dim_tag: Optional[Dim] = None, merged_dims: Optional[List[SizeValue]] = None,
              originating_tensor: Tensor = None):
    res = super(SizeValue, cls).__new__(cls, x)
    res.dim_tag = dim_tag or Dim(dimension=x, description="static_dim")
    res.merged_dims = merged_dims or []
    res.originating_tensor = originating_tensor
    return res

  @property
  def is_batch_dim(self):
    if self.dim_tag is None:
      return False
    return self.dim_tag.is_batch_dim()

  @property
  def originating_tensor_axis(self) -> int:
    assert self.originating_tensor is not None
    naming = Naming.get_instance()
    tensor_entry = naming.tensors[self.originating_tensor]
    returnn_axis = tensor_entry.returnn_data.get_axis_from_description(self.dim_tag)
    return tensor_entry.torch_axis_from_returnn_axis[returnn_axis]

  def get_originating_tensors(self) -> List[Tensor]:
    if self.originating_tensor is not None:
      return [self.originating_tensor]
    if self.merged_dims:
      return [
        d.originating_tensor
        for d in self.merged_dims
        if isinstance(d, SizeValue) and d.originating_tensor is not None]
    return []

  def as_tensor(self):
    if self.originating_tensor is None and self.merged_dims:
      tensor = numpy.prod([
        d.as_tensor() if isinstance(d, SizeValue) and d.dim_tag.dimension is None else int(d)
        for d in self.merged_dims])
    else:
      assert self.originating_tensor is not None
      from .nn.modules import Length
      tensor = Length(axis=self.originating_tensor_axis).as_returnn_torch_functional()(self.originating_tensor)
      if len(tensor.shape) > 0:
        from . import max
        tensor = max(tensor)
    tensor.fill_(int(self))
    tensor.is_defined = True
    naming = Naming.get_instance()
    tensor_entry = naming.tensors[tensor]
    tensor_entry.is_const = True
    tensor_entry.is_size_value = self
    return tensor

  def __repr__(self):
    res = super(SizeValue, self).__repr__()
    if self.dim_tag is None:
      return f"?{res}"
    return f"{self.dim_tag.short_repr()}({res})"

  def __mul__(self, other):
    assert isinstance(other, (int, SizeValue)), (  # could be allowed for static dims in the future
      "Multiplying a SizeValue with object of type {} is not allowed because it can lead to bugs, e.g. assumtion of a "
      "static batch dim.".format(type(other)))
    if type(other) == int and other == 1:
      return self
    merged_dims = [self, other]
    dim_tag = self.dim_tag * (other.dim_tag if isinstance(other, SizeValue) else other)
    return SizeValue(super(SizeValue, self).__mul__(other), dim_tag=dim_tag, merged_dims=merged_dims)

  def __rmul__(self, other):
    assert isinstance(other, (int, SizeValue)), (  # could be allowed for static dims in the future
      "Multiplying a SizeValue with object of type {} is not allowed because it can lead to bugs, e.g. assumtion of a "
      "static batch dim.".format(type(other)))
    if type(other) == int and other == 1:
      return self
    merged_dims = [other, self]
    dim_tag = (other.dim_tag if isinstance(other, SizeValue) else other) * self.dim_tag
    return SizeValue(super(SizeValue, self).__rmul__(other), dim_tag=dim_tag, merged_dims=merged_dims)


def zeros(*shape):
  from .tensor import Tensor
  return Tensor(*shape)


def empty(*shape):
  from .tensor import Tensor
  return Tensor(*shape)


def from_numpy(arr):
  import numpy
  if isinstance(arr, int):
    arr = numpy.array(arr, dtype='int32')
  if isinstance(arr, float):
    arr = numpy.array(arr, dtype='float32')
  if isinstance(arr, numpy.number):
    arr = numpy.array(arr)
  if isinstance(arr, (list, tuple)):
    _entry = from_numpy(arr[0])
    arr = numpy.array(arr, dtype=_entry.dtype.name)
  assert isinstance(arr, numpy.ndarray)
  from .tensor import Tensor
  return Tensor(*arr.shape, dtype=str(arr.dtype), numpy_array=arr)


def convert_to_tensor(x):
  import numpy
  from .tensor import Tensor
  from .nn.modules.operator import Stack
  if isinstance(x, Tensor):
    return x
  if isinstance(x, (int, float, numpy.number, numpy.ndarray)):
    return from_numpy(x)
  if isinstance(x, (list, tuple)):
    x = [convert_to_tensor(e) for e in x]
    return Stack(dim=0)(*x)
  raise TypeError(f"unexpected type {type(x)}")
