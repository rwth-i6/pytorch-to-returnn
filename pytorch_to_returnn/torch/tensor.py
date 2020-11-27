
from __future__ import annotations
from typing import Optional, Union
from functools import reduce
import operator
import numpy
from ._C import Size, dtype
from ..naming import Naming


_dtype = dtype


class Tensor:
  def __init__(self, *args, dtype: Optional[Union[str, _dtype]] = None, numpy_array: Optional[numpy.ndarray] = None):
    if dtype is not None and isinstance(dtype, _dtype):
      dtype = dtype.name
    if args and isinstance(args[0], Tensor):
      assert numpy_array is None
      numpy_array = args[0]._numpy_buffer.copy()
      shape = args[0].shape
    elif args and isinstance(args[0], (tuple, list)):
      shape = tuple(args[0])
    else:
      shape = args
    assert isinstance(shape, tuple) and all(isinstance(dim, int) for dim in shape)
    if numpy_array is not None:
      if dtype is not None:
        numpy_array = numpy_array.astype(dtype)
      else:
        dtype = str(numpy_array.dtype)
      assert numpy_array.shape == shape
    if dtype is None:
      dtype = "float32"
    self._shape = shape
    self._numpy_buffer = numpy.zeros(shape, dtype=dtype) if numpy_array is None else numpy_array
    self.dtype = _dtype(dtype)
    Naming.get_instance().register_tensor(self)

  def __repr__(self):
    return f"<Symbolic PyTorch {self.__class__.__name__} {self._shape}>"

  def dim(self):
    return len(self._shape)

  @property
  def ndim(self):
    return self.dim()

  def size(self, dim: Optional[int] = None):
    if dim is not None:
      return self._shape[dim]
    return Size(self._shape)

  @property
  def shape(self):
    return self.size()

  @property
  def data(self):
    return self  # TODO?

  def numel(self) -> int:
    """
    :returns: the total number of elements in the :attr:`input` tensor.
    """
    return reduce(operator.mul, self.shape, 1)

  def type(self, dtype=None, non_blocking=False, **kwargs) -> Union[_dtype, Tensor]:
    if dtype is None:
      return self.dtype
    from .nn.functional import cast
    return cast(self, dtype=dtype)

  def to(self, opt):
    return self  # ignore

  def view(self, *shape):
    if any(dim == -1 for dim in shape):
      num = self.numel()
      for dim in shape:
        if dim == -1:
          continue
        assert dim > 0 and num % dim == 0
        num //= dim
      shape = [dim if dim >= 0 else num for dim in shape]
    # TODO I need this in functional, as a real Module ...
    return Tensor(*shape, numpy_array=self._numpy_buffer.reshape(shape))

  def unsqueeze(self, dim: int):
    if dim < 0:
      dim += len(self._shape)
      assert dim >= 0
    return self.view(*(self._shape[:dim] + (-1,) + self._shape[dim:]))

  def copy_(self, source: Tensor):
    self._numpy_buffer = source.view(*self._shape).type(self.dtype)._numpy_buffer.copy()

  def normal_(self, mean=0, std=1):
    from .nn.init import normal_
    normal_(self, mean=mean, std=std)

  def detach(self):
    return self  # TODO use stop_gradient?

  def numpy(self):
    return self._numpy_buffer

  def float(self):
    from .nn.functional import cast
    return cast(self, "float32")

  def __getitem__(self, item):
    assert isinstance(item, int)  # not implemented otherwise
    assert self._shape  # cannot subscript a scalar
    return Tensor(*self._shape[1:])

  def __setitem__(self, key, value):
    self._numpy_buffer[key] = value

  def __add__(self, other):
    from .nn.functional import add
    return add(self, other)

  def __sub__(self, other):
    from .nn.functional import sub
    return sub(self, other)

  def __mul__(self, other):
    from .nn.functional import mul
    return mul(self, other)

  def __truediv__(self, other):
    from .nn.functional import truediv
    return truediv(self, other)
