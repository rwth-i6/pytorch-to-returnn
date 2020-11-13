
"""
Dummies...
"""


def device(name):
  return name


class dtype(str):
  pass


class Size(tuple):
  pass


def zeros(*shape):
  from .tensor import Tensor
  return Tensor(*shape)


def from_numpy(arr):
  import numpy
  assert isinstance(arr, numpy.ndarray)
  from .tensor import Tensor
  return Tensor(*arr.shape)
