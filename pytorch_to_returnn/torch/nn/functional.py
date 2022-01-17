
"""
This wraps both torch.nn.functional and torch.functional, and other __torch_function__ on tensors.

Note that this all maps to modules, which will be temporarily created.
In RETURNN, every operation is a layer.
"""

import numpy
import warnings
from pytorch_to_returnn import torch
from typing import Optional, Union, List, Tuple, Dict, TypeVar, Sequence
from . import modules
from ..tensor import Tensor
from .._C import Size, dtype as _dtype
from ...naming import Naming


_number = Union[int, float, numpy.ndarray, numpy.number]
_size = Union[Size, List[int], Tuple[int, ...]]
_shape_t = Union[int, List[int], Size]
_T = TypeVar("_T")
_default_float_type = "float32"
_builtin_sum = sum


def zeros(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False):
  return Tensor(*size, dtype=dtype)


def ones(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False):
  if size and isinstance(size[0], (tuple, list)):
    assert len(size) == 1
    size = tuple(size[0])
  if not dtype:
    dtype = _default_float_type
  return tensor(numpy.ones(size, dtype=dtype))


def arange(*args, out: Optional[Tensor]=None, dtype: Optional[_dtype]=None, device=None,
           requires_grad: bool=False) -> Tensor:
  assert 1 <= len(args) <= 3
  if len(args) == 3:
    start, end, step = args
  elif len(args) == 2:
    start, end = args
    step = 1
  elif len(args) == 1:
    end, = args
    start = 0
    step = 1
  mod = modules.Range()
  return mod(end, start, step, dtype, False)


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


def is_tensor(obj) -> bool:
  return isinstance(obj, Tensor)


def sum(input: Tensor, dim: Optional[int] = None, dtype: Optional[Union[str, _dtype]] = None) -> Tensor:
  assert dim is not None, "not implemented yet"
  return modules.Reduce(mode="sum", axes=dim)(input)


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


def unsqueeze(input: Tensor, dim: int) -> Tensor:
  return input.unsqueeze(dim)


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


def permute(input: Tensor, dims: Tuple[int]):
  return tensorflow_transpose(input, perm=dims)


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
  assert _builtin_sum(size_splits) == input.shape[dim] and len(size_splits) == chunks
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


def gelu(input: Tensor) -> Tensor:
  return modules.GELU().as_returnn_torch_functional()(input)


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


def log1p(input, *, out=None):
  return log(input + 1)


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


def layer_norm(input: Tensor, normalized_shape: _shape_t, weight: Optional[Tensor] = None,
               bias: Optional[Tensor] = None, eps: float = 1e-5) -> Tensor:
  module = modules.LayerNorm(normalized_shape=normalized_shape, eps=eps, elementwise_affine=False)
  out = module.as_returnn_torch_functional()(input)
  if weight is not None:
    assert (out.shape[-1],) == weight.shape, "data should be of shape (B, *, F) and weights should be of shape (F,)"
    out *= weight
  if bias is not None:
    assert (out.shape[-1],) == bias.shape, "data should be of shape (B, *, F) and bias should be of shape (F,)"
    out += bias
  return out


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


def multi_head_attention_forward(
  query: Tensor,
  key: Tensor,
  value: Tensor,
  embed_dim_to_check: int,
  num_heads: int,
  in_proj_weight: Tensor,
  in_proj_bias: Optional[Tensor],
  bias_k: Optional[Tensor],
  bias_v: Optional[Tensor],
  add_zero_attn: bool,
  dropout_p: float,
  out_proj_weight: Tensor,
  out_proj_bias: Optional[Tensor],
  training: bool = True,
  key_padding_mask: Optional[Tensor] = None,
  need_weights: bool = True,
  attn_mask: Optional[Tensor] = None,
  use_separate_proj_weight: bool = False,
  q_proj_weight: Optional[Tensor] = None,
  k_proj_weight: Optional[Tensor] = None,
  v_proj_weight: Optional[Tensor] = None,
  static_k: Optional[Tensor] = None,
  static_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
  # pytorch-to-returnn: remove possible `return handle_torch_function(...)` from original implementation

  tgt_len, bsz, embed_dim = query.size()
  assert embed_dim == embed_dim_to_check
  # allow MHA to have different sizes for the feature dimension
  assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

  if isinstance(embed_dim, torch.Tensor):
    # embed_dim can be a tensor when JIT tracing
    head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
  else:
    head_dim = embed_dim // num_heads
  assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
  scaling = float(head_dim) ** -0.5

  if not use_separate_proj_weight:
    if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
      # self-attention
      q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

    elif key is value or torch.equal(key, value):
      # encoder-decoder attention
      # This is inline in_proj function with in_proj_weight and in_proj_bias
      _b = in_proj_bias
      _start = 0
      _end = embed_dim
      _w = in_proj_weight[_start:_end, :]
      if _b is not None:
        _b = _b[_start:_end]
      q = linear(query, _w, _b)

      if key is None:
        assert value is None
        k = None
        v = None
      else:

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = embed_dim
        _end = None
        _w = in_proj_weight[_start:, :]
        if _b is not None:
          _b = _b[_start:]
        k, v = linear(key, _w, _b).chunk(2, dim=-1)

    else:
      # This is inline in_proj function with in_proj_weight and in_proj_bias
      _b = in_proj_bias
      _start = 0
      _end = embed_dim
      _w = in_proj_weight[_start:_end, :]
      if _b is not None:
        _b = _b[_start:_end]
      q = linear(query, _w, _b)

      # This is inline in_proj function with in_proj_weight and in_proj_bias
      _b = in_proj_bias
      _start = embed_dim
      _end = embed_dim * 2
      _w = in_proj_weight[_start:_end, :]
      if _b is not None:
        _b = _b[_start:_end]
      k = linear(key, _w, _b)

      # This is inline in_proj function with in_proj_weight and in_proj_bias
      _b = in_proj_bias
      _start = embed_dim * 2
      _end = None
      _w = in_proj_weight[_start:, :]
      if _b is not None:
        _b = _b[_start:]
      v = linear(value, _w, _b)
  else:
    q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
    len1, len2 = q_proj_weight_non_opt.size()
    assert len1 == embed_dim and len2 == query.size(-1)

    k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
    len1, len2 = k_proj_weight_non_opt.size()
    assert len1 == embed_dim and len2 == key.size(-1)

    v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
    len1, len2 = v_proj_weight_non_opt.size()
    assert len1 == embed_dim and len2 == value.size(-1)

    if in_proj_bias is not None:
      q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
      k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim: (embed_dim * 2)])
      v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
    else:
      q = linear(query, q_proj_weight_non_opt, in_proj_bias)
      k = linear(key, k_proj_weight_non_opt, in_proj_bias)
      v = linear(value, v_proj_weight_non_opt, in_proj_bias)
  q = q * scaling

  if attn_mask is not None:
    assert (
      attn_mask.dtype == torch.float32
      or attn_mask.dtype == torch.float64
      or attn_mask.dtype == torch.float16
      or attn_mask.dtype == torch.uint8
      or attn_mask.dtype == torch.bool
    ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
    if attn_mask.dtype == torch.uint8:
      warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
      attn_mask = attn_mask.to(torch.bool)

    if attn_mask.dim() == 2:
      attn_mask = attn_mask.unsqueeze(0)
      if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
        raise RuntimeError("The size of the 2D attn_mask is not correct.")
    elif attn_mask.dim() == 3:
      if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
        raise RuntimeError("The size of the 3D attn_mask is not correct.")
    else:
      raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
    # attn_mask's dim is 3 now.

  # convert ByteTensor key_padding_mask to bool
  if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
    warnings.warn(
      "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
    )
    key_padding_mask = key_padding_mask.to(torch.bool)

  if bias_k is not None and bias_v is not None:
    if static_k is None and static_v is None:
      k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
      v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
      if attn_mask is not None:
        attn_mask = pad(attn_mask, (0, 1))
      if key_padding_mask is not None:
        key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
      assert static_k is None, "bias cannot be added to static key."
      assert static_v is None, "bias cannot be added to static value."
  else:
    assert bias_k is None
    assert bias_v is None

  # Unlike the original implementation, we keep batch axis and num_heads separate.
  # Therefore, reshape q, k and v to (bsz, num_heads, tgt_len, head_dim)
  q = q.contiguous().view(tgt_len, bsz, num_heads, head_dim).transpose(0, 1).transpose(1, 2)
  if k is not None:
    k = k.contiguous().view(-1, bsz, num_heads, head_dim).transpose(0, 1).transpose(1, 2)
  if v is not None:
    v = v.contiguous().view(-1, bsz, num_heads, head_dim).transpose(0, 1).transpose(1, 2)

  if static_k is not None:
    assert static_k.size(0) == bsz * num_heads
    assert static_k.size(2) == head_dim
    k = static_k.view(bsz, num_heads, -1, head_dim)  # adapt to separate batch axis

  if static_v is not None:
    assert static_v.size(0) == bsz * num_heads
    assert static_v.size(2) == head_dim
    v = static_v.view(bsz, num_heads, -1, head_dim)  # adapt to separate batch axis

  src_len = k.size(2)  # adapt to separate batch axis

  if key_padding_mask is not None:
    assert key_padding_mask.size(0) == bsz
    assert key_padding_mask.size(1) == src_len

  if add_zero_attn:
    src_len += 1
    # adapt to separate batch axis
    k = torch.cat([k, torch.zeros(k.size()[:2] + (1, k.size(3)), dtype=k.dtype, device=k.device)], dim=2)
    v = torch.cat([v, torch.zeros(v.size()[:2] + (1, v.size(3)), dtype=v.dtype, device=v.device)], dim=2)
    if attn_mask is not None:
      attn_mask = pad(attn_mask, (0, 1))
    if key_padding_mask is not None:
      key_padding_mask = pad(key_padding_mask, (0, 1))

  # adapt to separate batch axis
  attn_output_weights = torch.matmul(q, k.transpose(2, 3))
  assert list(attn_output_weights.size()) == [bsz, num_heads, tgt_len, src_len]

  if attn_mask is not None:
    # adapt to separate batch axis
    attn_mask = attn_mask.view(bsz, num_heads, tgt_len, src_len)
    if attn_mask.dtype == torch.bool:
      attn_output_weights.masked_fill_(attn_mask, float("-inf"))
    else:
      attn_output_weights += attn_mask

  if key_padding_mask is not None:
    # adapt to separate batch axis
    attn_output_weights = attn_output_weights.masked_fill(
      key_padding_mask.unsqueeze(1).unsqueeze(2),
      float("-inf"),
    )

  attn_output_weights = softmax(attn_output_weights, dim=-1)
  attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

  # adapt to separate batch axis
  attn_output = torch.matmul(attn_output_weights, v)
  assert list(attn_output.size()) == [bsz, num_heads, tgt_len, head_dim]
  attn_output = attn_output.transpose(1, 2).transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
  attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

  if need_weights:
    # average attention weights over heads
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    return attn_output, attn_output_weights.sum(dim=1) / num_heads
  else:
    return attn_output, None


def mse_loss():
  raise NotImplementedError


def ceil():
  raise NotImplementedError


def clamp():
  raise NotImplementedError


def exp():
  raise NotImplementedError


def log10():
  raise NotImplementedError
