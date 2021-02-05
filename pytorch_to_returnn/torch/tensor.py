
from __future__ import annotations
from typing import Optional, Union, List
from functools import reduce
import operator
import numpy
from ._C import Size, SizeValue, dtype
from ..naming import Naming


_dtype = dtype


class Tensor:
  def __init__(self, *args, dtype: Optional[Union[str, _dtype]] = None, numpy_array: Optional[numpy.ndarray] = None):
    if dtype is not None and isinstance(dtype, _dtype):
      dtype = dtype.name
    if args and isinstance(args[0], Tensor):
      assert len(args) == 1
      assert numpy_array is None
      numpy_array = args[0]._numpy_buffer.copy()
      shape = args[0].shape
    elif args and isinstance(args[0], (tuple, list)):
      assert len(args) == 1
      shape = tuple(args[0])
    else:
      shape = args
    assert isinstance(shape, tuple) and all(isinstance(dim, int) for dim in shape)
    shape = tuple([d if isinstance(d, SizeValue) else SizeValue(d) for d in shape])
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
    return f"<{self.__class__.__name__} {self.returnn_naming_entry.repr_content()}>"

  @property
  def returnn_naming_entry(self):
    return Naming.get_instance().register_tensor(self)

  def dim(self):
    return len(self._shape)

  @property
  def ndim(self):
    return self.dim()

  def size(self, dim: Optional[int] = None) -> Size[SizeValue, ...]:
    if dim is not None:
      return self._shape[dim]
    return Size(self._shape)

  @property
  def shape(self) -> Size[SizeValue, ...]:
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

  def type_as(self, tensor) -> Tensor:
    return self.type(tensor.type())

  def to(self, opt):
    return self  # ignore

  def contiguous(self):
    return self  # ignore

  def clone(self):
    return self  # ignore

  def device(self):
    return None  # ignore

  def flatten(self):
    from .nn.functional import flatten
    return flatten(self)

  def view(self, *shape):
    from .nn.functional import reshape
    if shape and isinstance(shape[0], (tuple, list)):
      assert len(shape) == 1
      shape = tuple(shape[0])
    assert isinstance(shape, tuple)
    return reshape(self, shape)

  def unsqueeze(self, dim: int):
    assert -len(self._shape) <= dim <= len(self._shape)
    if dim < 0:
      dim += len(self._shape) + 1
    assert 0 <= dim <= len(self._shape)
    return self.view(*(self._shape[:dim] + (1,) + self._shape[dim:]))

  def transpose(self, dim0: int, dim1: int):
    from .nn.functional import transpose
    return transpose(self, dim0=dim0, dim1=dim1)

  def t(self):
    from .nn.functional import t
    return t(self)

  def expand(self, *sizes):
    from .nn.functional import expand
    if sizes and isinstance(sizes[0], (tuple, list)):
      assert len(sizes) == 1
      sizes = tuple(sizes[0])
    assert isinstance(sizes, tuple)
    return expand(self, sizes)

  def resize_(self, *sizes, memory_format=None):
    # memory format is ignored
    assert len(self.shape) == 0  # not implemented otherwise
    self._shape = sizes
    self._numpy_buffer = numpy.zeros(sizes, dtype=self.dtype.canonical_name)

  def new_zeros(self, *size, dtype=None, device=None, requires_grad=False):
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
      size = size[0]
    if dtype is None:
      dtype = self.dtype
    else:
      dtype = _dtype(dtype)
    from .nn.functional import zeros
    return zeros(size, dtype=dtype)

  def new_empty(self, *size, dtype=None, device=None, requires_grad=False):
    # use new_zeros here to avoid errors by uninitialized memory
    return self.new_zeros(size, dtype, device, requires_grad)

  def new(self, *args, dtype=None, device=None, requires_grad=False):
    if args and isinstance(args[0], Tensor):
      assert len(args) == 1
      assert args[0].type() == self.type()
      return args[0]
    else:
      shape = args
      return self.new_empty(shape, dtype, device, requires_grad)

  def copy_(self, source: Tensor):
    self._numpy_buffer = source.view(*self._shape).type(self.dtype)._numpy_buffer.copy()

  def normal_(self, mean=0, std=1):
    from .nn.init import normal_
    normal_(self, mean=mean, std=std)
    return self

  def uniform_(self, a=0, b=1):
    from .nn.init import uniform_
    uniform_(self, a=a, b=b)
    return self

  def zero_(self):
    self.fill_(0)
    return self

  def fill_(self, x):
    if not self._shape:  # scalar
      self._numpy_buffer = numpy.array(x, dtype=self.dtype.name)
    else:
      self._numpy_buffer[:] = x

  def detach(self):
    return self  # TODO use stop_gradient?

  def numpy(self):
    return self._numpy_buffer

  def float(self):
    from .nn.functional import cast
    return cast(self, "float32")

  def int(self):
    from .nn.functional import cast
    return cast(self, "int32")

  def abs(self):
    from .nn.functional import abs
    return abs(self)

  def log(self):
    from .nn.functional import log
    return log(self)

  def sigmoid(self):
    from .nn.functional import sigmoid
    return sigmoid(self)

  def pow(self, exponent: float):
    from .nn.functional import pow
    return pow(self, exponent)

  def chunk(self, chunks: int, dim: int = 0) -> List[Tensor]:
    from .nn.functional import chunk
    return chunk(self, chunks=chunks, dim=dim)

  def matmul(self, tensor2: Tensor) -> Tensor:
    from .nn.functional import matmul
    return matmul(self, tensor2)

  def __getitem__(self, item):
    assert self._shape  # cannot subscript a scalar
    if isinstance(item, int):
      from .nn import Gather
      return Gather(dim=0, pos=item)(self)
    elif isinstance(item, slice):
      from .nn import Slice
      return Slice(axis=0, start=item.start, stop=item.stop, step=item.step)(self)
    elif isinstance(item, tuple):
      assert len(item) == self.ndim
      from .nn import Slice
      out = self
      for ax, ax_slice in enumerate(item):
        assert isinstance(ax_slice, slice)
        if not ax_slice.start and not ax_slice.stop and ax_slice.step in {1, None}:
          continue
        out = Slice(axis=ax, start=ax_slice.start, stop=ax_slice.stop, step=ax_slice.step)(out)
      return out
    else:
      raise NotImplementedError

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

  def __radd__(self, other):
    from .nn.functional import add
    return add(other, self)

  def __rsub__(self, other):
    from .nn.functional import sub
    return sub(other, self)

  def __rmul__(self, other):
    from .nn.functional import mul
    return mul(other, self)

  def __rtruediv__(self, other):
    from .nn.functional import truediv
    return truediv(other, self)

  def __ge__(self, other):
    from .nn.functional import greater_equal
    return greater_equal(self, other)


class LongTensor(Tensor):
  def __init__(self, *args):
    super(LongTensor, self).__init__(dtype="int64", *args)


class FloatTensor(Tensor):
  def __init__(self, *args):
    super(FloatTensor, self).__init__(dtype="float32", *args)
