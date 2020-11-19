
"""
Dummies...
"""

from __future__ import annotations
from typing import Union


def device(name):
  return name


# noinspection PyPep8Naming
class dtype:
  def __init__(self, arg: Union[str, dtype]):
    if isinstance(arg, str):
      self.name = arg
    elif isinstance(arg, dtype):
      self.name = arg.name
    else:
      raise TypeError(f"unexpected arg {arg}")

  def __str__(self):
    return f"torch.{self.name}"


class Size(tuple):
  pass


def zeros(*shape):
  from .tensor import Tensor
  return Tensor(*shape)


def from_numpy(arr):
  import numpy
  if isinstance(arr, int):
    arr = numpy.int32(arr)
  if isinstance(arr, float):
    arr = numpy.float(arr)
  if isinstance(arr, numpy.number):
    arr = numpy.array(arr)
  assert isinstance(arr, numpy.ndarray)
  from .tensor import Tensor
  return Tensor(*arr.shape, dtype=str(arr.dtype))
