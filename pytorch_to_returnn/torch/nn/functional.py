
"""
This wraps both torch.nn.functional and torch.functional, and other __torch_function__ on tensors.

Note that this all maps to modules, which will be temporarily created.
In RETURNN, every operation is a layer.
"""

import numpy
from typing import Optional, Union, List, Tuple, Dict, TypeVar, Sequence
from . import modules
from ..tensor import Tensor
from .._C import Size, dtype as _dtype
from ...naming import Naming


_number = Union[int, float, numpy.ndarray, numpy.number]
_size = Union[Size, List[int], Tuple[int, ...]]
_T = TypeVar("_T")
_default_float_type = "float32"


def zeros(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False):
  return Tensor(*size, dtype=dtype)


def ones(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False):
  if size and isinstance(size[0], (tuple, list)):
    assert len(size) == 1
    size = tuple(size[0])
  if not dtype:
    dtype = _default_float_type
  return tensor(numpy.ones(size, dtype=dtype))


def tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False):
  from .._C import from_numpy
  x = from_numpy(data)
  if dtype:
    x = x.type(dtype)
  return x


def cast(input: Union[_T, Tensor, _number], dtype: Union[str, _dtype]) -> Union[_T, Tensor]:
  dtype = _dtype(dtype)
  if dtype == get_dtype(input):
    return input
  return modules.Cast(dtype=dtype)(input)

def cat(tensors, dim=0):
  from .modules.operator import Cat
  return Cat(dim).as_returnn_torch_functional()(*tensors)


def get_dtype(tensor: Union[Tensor, _number]) -> _dtype:
  if isinstance(tensor, Tensor):
    return tensor.dtype
  if isinstance(tensor, int):
    return _dtype("int32")
  if isinstance(tensor, float):
    return _dtype(_default_float_type)
  if isinstance(tensor, (numpy.number, numpy.ndarray)):
    return _dtype(str(tensor.dtype))
  raise TypeError(f"unexpected type {type(tensor)}")


def result_type(tensor1: Union[Tensor, _number], tensor2: Union[Tensor, _number]) -> _dtype:
  # https://pytorch.org/docs/stable/generated/torch.result_type.html
  return promote_types(get_dtype(tensor1), get_dtype(tensor2))


def promote_types(type1: Union[str, _dtype], type2: Union[str, _dtype]) -> _dtype:
  # https://pytorch.org/docs/stable/generated/torch.promote_types.html
  type1 = _dtype(type1)
  type2 = _dtype(type2)
  if type1.category_int != type2.category_int:
    if type1.category_int < type2.category_int:
      type1, type2 = type2, type1
    assert type1.category_int > type2.category_int
    return type1
  assert type1.category_int == type2.category_int
  if type1.bit_size == type2.bit_size:
    assert type1 == type2
    return type1
  if type1.bit_size < type2.bit_size:
    type1, type2 = type2, type1
  assert type1.bit_size > type2.bit_size
  return type1


def as_tensor(data: Union[Tensor, _number],
              dtype: Optional[Union[str, _dtype]] = None,
              device=None) -> Tensor:
  if not isinstance(data, Tensor):
    from .._C import from_numpy
    data = from_numpy(data)
  assert isinstance(data, Tensor)
  if dtype:
    data = cast(data, dtype)
  return data


def add(x: Tensor, y: Tensor) -> Tensor:
  dtype = result_type(x, y)
  return modules.BinaryOperator(kind="add")(cast(x, dtype), cast(y, dtype))


def sub(x: Tensor, y: Tensor) -> Tensor:
  dtype = result_type(x, y)
  return modules.BinaryOperator(kind="sub")(cast(x, dtype), cast(y, dtype))


def mul(x: Tensor, y: Tensor) -> Tensor:
  dtype = result_type(x, y)
  return modules.BinaryOperator(kind="mul")(cast(x, dtype), cast(y, dtype))


def matmul(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
  assert out is None, "not implemented otherwise"
  mod = modules.Matmul()
  return mod(input, other)


def bmm(input: Tensor, mat2: Tensor, *, deterministic: bool = False, out: Optional[Tensor] = None) -> Tensor:
  assert out is None, "not implemented otherwise"
  assert input.shape[0] == mat2.shape[0], "First axis must match"
  assert input.shape[2] == mat2.shape[1], "Axes for matrix-matrix product must match"
  return matmul(input, mat2)


def truediv(x: Tensor, y: Tensor) -> Tensor:
  dtype = result_type(x, y)
  return modules.BinaryOperator(kind="truediv")(cast(x, dtype), cast(y, dtype))


def greater_equal(x: Tensor, y: Tensor) -> Tensor:
  dtype = result_type(x, y)
  return modules.ComparisonOperator(kind="greater_equal")(cast(x, dtype), cast(y, dtype))


def flatten(input: Tensor, start_dim=0, end_dim=-1) -> Tensor:
  return modules.Flatten(start_dim=start_dim, end_dim=end_dim).as_returnn_torch_functional()(input)


def reshape(input: Tensor, shape: Tuple[int, ...]) -> Tensor:
  if any(dim == -1 for dim in shape):
    num = input.numel()
    for dim in shape:
      if dim == -1:
        continue
      assert dim > 0 and num % dim == 0
      num //= dim
    shape = [dim if dim >= 0 else num for dim in shape]

  # Use Flatten, Unflatten, Squeeze.
  # (Other reshapes are disallowed.)
  axis1, axis2 = 0, 0
  while axis1 < len(input.shape) and axis2 < len(shape):
    if input.shape[axis1] == shape[axis2]:
      axis1 += 1
      axis2 += 1
      continue
    elif input.shape[axis1] < shape[axis2]:
      if input.shape[axis1] == 1:
        input = modules.Squeeze(dim=axis1).as_returnn_torch_functional()(input)
        continue
      n = 1
      a = axis1
      while a < len(input.shape) and n < shape[axis2]:
        assert shape[axis2] % n == 0 and n < shape[axis2]
        n *= input.shape[a]
        a += 1
      if n == shape[axis2]:
        input = modules.Flatten(start_dim=axis1, end_dim=a - 1).as_returnn_torch_functional()(input)
        assert input.shape[axis1] == shape[axis2]
        continue
      elif shape[axis2] % input.shape[axis1] == 0:
        split_factor = shape[axis2] // input.shape[axis1]
        assert input.shape[axis1 + 1] == shape[axis2 + 1] * split_factor
        # something like (..., a, b*c,...) -> (..., a*b, c,...)
        unflattened_size = (split_factor, shape[axis2 + 1])
        input = modules.Unflatten(dim=axis1 + 1, unflattened_size=unflattened_size).as_returnn_torch_functional()(input)
        input = modules.Flatten(start_dim=axis1, end_dim=axis1 + 1).as_returnn_torch_functional()(input)
        assert input.shape[axis1] == shape[axis2]
        axis1 += 1
        axis2 += 1
        assert input.shape[axis1] == shape[axis2]
        continue
    elif input.shape[axis1] > shape[axis2]:
      n = 1
      a = axis2
      while a < len(shape) and n < input.shape[axis1]:
        assert input.shape[axis1] % n == 0 and n < input.shape[axis1]
        n *= shape[a]
        a += 1
      if n == input.shape[axis1]:
        unflattened_size = tuple(shape[axis2:a])
        input = modules.Unflatten(dim=axis1, unflattened_size=unflattened_size).as_returnn_torch_functional()(input)
        assert input.shape[axis1] == shape[axis2]
        continue
      elif input.shape[axis1] % shape[axis2] == 0:
        split_factor = input.shape[axis1] // shape[axis2]
        assert input.shape[axis1 + 1] * split_factor == shape[axis2 + 1]
        # something like (..., a*b, c,...) -> (..., a, b*c,...)
        unflattened_size = (shape[axis2], split_factor)
        input = modules.Unflatten(dim=axis1, unflattened_size=unflattened_size).as_returnn_torch_functional()(input)
        assert input.shape[axis1] == shape[axis2]
        axis1 += 1
        axis2 += 1
        input = modules.Flatten(start_dim=axis1, end_dim=axis1 + 1).as_returnn_torch_functional()(input)
        assert input.shape[axis1] == shape[axis2]
        continue
    else:
      assert False  # cannot happen
  assert axis1 == axis2
  if len(input.shape) < len(shape):
    assert all(shape[i] == 1 for i in range(len(input.shape), len(shape)))
    input = modules.Unflatten(
      dim=-1, unflattened_size=shape[len(input.shape) - 1:]).as_returnn_torch_functional()(input)
  elif len(input.shape) > len(shape):
    while len(input.shape) > len(shape):
      input = modules.Squeeze(dim=len(shape)).as_returnn_torch_functional()(input)
  assert len(input.shape) == len(shape) and input.shape == tuple(shape)
  return input


def movedim(input: Tensor, source: Union[int, Tuple[int, ...]], destination: Union[int, Tuple[int, ...]]):
  if isinstance(source, int):
    source = (source,)
  if isinstance(destination, int):
    destination = (destination,)
  assert isinstance(source, (tuple, list)) and isinstance(destination, (tuple, list))
  assert len(source) == len(destination)
  perm = {i: j for i, j in zip(destination, source)}
  # All remaining axes stay in order.
  return tensorflow_transpose(input, perm=perm)


def transpose(input: Tensor, dim0: int, dim1: int):
  return tensorflow_transpose(input, perm={dim0: dim1, dim1: dim0})


def t(input: Tensor):
  if len(input.shape) < 2:
    return input
  elif len(input.shape) == 2:
    return transpose(input, 0, 1)
  else:
    # https://pytorch.org/docs/stable/generated/torch.t.html#torch.t
    raise ValueError("t() expects input to be <= 2-D tensor")


def tensorflow_transpose(input: Tensor, perm: Optional[Union[Dict[int, int], Tuple[int, ...], List[int]]]):
  """
  Note: This function is added by us, not available in original PyTorch.

  Note: The resulting Torch tensor is transposed as expected.
  However, on the RETURNN side, we actually should never need to transpose,
  as we have dimension tags, and all layers should refer to axes by dim tags.
  So on RETURNN side, this is a no-op.
  """
  return modules.Transpose(perm=perm)(input)


def expand(input: Tensor, *sizes: _size):
  if sizes and isinstance(sizes[0], (tuple, list)):
    assert len(sizes) == 1
    shape = tuple(sizes[0])
  else:
    shape = sizes
  assert isinstance(shape, tuple)

  # reshape to add new dims
  if len(shape) != len(input.shape):
    assert len(shape) >= len(input.shape), "Cannot reduce dimensions"
    assert -1 not in [d for d in shape[:-len(input.shape)]], "Size -1 is not allowed for expanded dims"
    exp_shape = (1,) * (len(shape) - len(input.shape)) + input.shape[-len(input.shape):]
    input = reshape(input, shape=exp_shape)

  # tile to expand to larger size
  multiples = {}
  for axis, multiple in enumerate(shape):
    if input.shape[axis] > 1:
      assert multiple == -1 or multiple == input.shape[axis], \
        f"Expanded size ({multiple}) must match existing size ({input.shape[axis]}) at non-singleton dimension ({axis})"
      multiple = 1
    if multiple == -1:
      multiple = 1  # -1 as the size for a dim means not changing the size of that dim
    multiples[axis] = multiple
  return modules.Tile(multiples=multiples).as_returnn_torch_functional()(input)


def chunk(input: Tensor, chunks: int, dim: int = 0) -> List[Tensor]:
  chunk_size = input.shape[dim] // chunks
  if input.shape[dim] % chunks != 0:
    chunk_size += 1
    size_splits = [chunk_size] * chunks
    size_splits[-1] -= chunk_size * chunks - input.shape[dim]
  else:
    size_splits = [chunk_size] * chunks
  assert sum(size_splits) == input.shape[dim] and len(size_splits) == chunks
  return modules.Split(dim=dim, size_splits=size_splits).as_returnn_torch_functional()(input)


def pad(input: Tensor, pad, mode='constant', value=0) -> Tensor:
  return modules.GenericPadNd(padding=pad, mode=mode, value=value).as_returnn_torch_functional()(input)


def max(*inputs: Tensor) -> Tensor:
  return modules.Max()(*inputs)


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None):
  output = input.matmul(weight.t())
  if bias is not None:
    output += bias
  return output


def conv1d(
    input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
    stride: Union[int, _size] = 1, padding: Union[int, _size] = 0,
    dilation: Union[int, _size] = 1, groups: int = 1) -> Tensor:
  mod = modules.FunctionalConv1d(stride=stride, padding=padding, dilation=dilation, groups=groups)
  return mod(input, weight, bias)


def conv2d(
    input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
    stride: Union[int, _size] = 1, padding: Union[int, _size] = 0,
    dilation: Union[int, _size] = 1, groups: int = 1) -> Tensor:
  mod = modules.FunctionalConv2d(stride=stride, padding=padding, dilation=dilation, groups=groups)
  return mod(input, weight, bias)


def conv_transpose1d(
    input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
    stride: Union[int, _size] = 1, padding: Union[int, _size] = 0,
    output_padding: Union[int, _size] = 0,
    groups: int = 1,
    dilation: Union[int, _size] = 1) -> Tensor:
  mod = modules.FunctionalConvTransposed1d(
    stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups)
  return mod(input, weight, bias)


def max_pool2d(input: Tensor, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
  mod = modules.MaxPool2d(
    kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
    ceil_mode=ceil_mode, return_indices=return_indices)
  mod.as_returnn_torch_functional()
  return mod(input)


def relu(input: Tensor) -> Tensor:
  return modules.ReLU().as_returnn_torch_functional()(input)


def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
  return modules.LeakyReLU(negative_slope=negative_slope, inplace=inplace).as_returnn_torch_functional()(input)


def sqrt(input: Tensor) -> Tensor:
  return modules.Sqrt().as_returnn_torch_functional()(input)


def tanh(input: Tensor) -> Tensor:
  return modules.Tanh().as_returnn_torch_functional()(input)


def softmax(input: Tensor, dim: Optional[int] = None, dtype=None):
  return modules.Softmax(dim=dim).as_returnn_torch_functional()(input)


def abs(input: Tensor) -> Tensor:
  return modules.Abs().as_returnn_torch_functional()(input)


def log_softmax(input: Tensor, dim: Optional[int] = None, dtype=None):
  return modules.LogSoftmax(dim=dim).as_returnn_torch_functional()(input)


def log(input: Tensor):
  return modules.Log().as_returnn_torch_functional()(input)


def sigmoid(input: Tensor):
  return modules.Sigmoid().as_returnn_torch_functional()(input)


def logsigmoid(input: Tensor):
  return modules.LogSigmoid().as_returnn_torch_functional()(input)


def pow(input: Tensor, exponent: float):
  return modules.Power(exponent=exponent).as_returnn_torch_functional()(input)


def normalize(input: Tensor, p=2, dim=1, eps=1e-12) -> Tensor:
  norm_ = modules.Norm(p=p, axes=[dim], keepdims=True)(input)
  norm_f = modules.Reciprocal(eps=eps)(norm_)
  return input * norm_f


def norm(input: Tensor,
         p: Optional[Union[str, float, int]] = "fro",
         dim: Optional[Union[int, List[int]]] = None,
         keepdim: bool = False) -> Tensor:
  return modules.Norm(p=p, axes=[dim], keepdims=keepdim)(input)


def norm_except_dim(v: Tensor, pow: int = 2, dim: int = 0) -> Tensor:
  return modules.Norm(p=pow, axes=[i for i in range(v.dim()) if i != dim], keepdims=True)(v)


def group_norm(input: Tensor, num_groups: int, weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
               eps: float = 1e-5) -> Tensor:
  module = modules.GroupNorm(num_groups=num_groups, num_channels=input.shape[1], eps=eps, affine=False)
  out = module.as_returnn_torch_functional()(input)
  if weight is not None:
    assert (out.shape[1],) == weight.shape, "data should be of shape (B, F, *) and weights should be of shape (F,)"
    out *= weight.view(1, -1, 1)
  if bias is not None:
    assert (out.shape[1],) == bias.shape, "data should be of shape (B, F, *) and bias should be of shape (F,)"
    out += bias.view(1, -1, 1)
  return out


def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
  if p < 0. or p > 1.:
    raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
  if p == 0.0 or not training:
    return input
  if p > 0.0:
    return modules.Dropout(p=p, inplace=inplace).as_returnn_torch_functional()(input)
