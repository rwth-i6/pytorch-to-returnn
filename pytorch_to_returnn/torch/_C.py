
"""
Dummies...
"""

from __future__ import annotations
from typing import Union, Any, Tuple


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
  We extend this, to store extra information, e.g. such that this reflects the batch-dim.
  """
  is_batch_dim: bool = False

  def __repr__(self):
    res = super(SizeValue, self).__repr__()
    if self.is_batch_dim:
      res = f"Batch({res})"
    return res


def zeros(*shape):
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
  assert isinstance(arr, numpy.ndarray)
  from .tensor import Tensor
  return Tensor(*arr.shape, dtype=str(arr.dtype), numpy_array=arr)
