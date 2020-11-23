
"""
This wraps both torch.nn.functional and torch.functional, and other __torch_function__ on tensors.

Note that this all maps to modules, which will be temporarily created.
In RETURNN, every operation is a layer.
"""

from typing import Optional, Union, List, Tuple, Dict
from . import modules
from ..tensor import Tensor
from .._C import Size


_size = Union[Size, List[int], Tuple[int, ...]]


def add(x: Tensor, y: Tensor) -> Tensor:
  return modules.BinaryOperator(kind="add")(x, y)


def sub(x: Tensor, y: Tensor) -> Tensor:
  return modules.BinaryOperator(kind="sub")(x, y)


def mul(x: Tensor, y: Tensor) -> Tensor:
  return modules.BinaryOperator(kind="mul")(x, y)


def truediv(x: Tensor, y: Tensor) -> Tensor:
  return modules.BinaryOperator(kind="truediv")(x, y)


def movedim(input: Tensor, source: Union[int, Tuple[int, ...]], destination: Union[int, Tuple[int, ...]]):
  if isinstance(source, int):
    source = (source,)
  if isinstance(destination, int):
    destination = (destination,)
  assert isinstance(source, (tuple, list)) and isinstance(destination, (tuple, list))
  assert len(source) == len(destination)
  return tensorflow_transpose(input, perm={i: j for (i, j) in zip(destination, source)})


def transpose(input: Tensor, dim0: int, dim1: int):
  return tensorflow_transpose(input, perm={dim0: dim1, dim1: dim0})


def tensorflow_transpose(input: Tensor, perm: Optional[Union[Dict[int, int], Tuple[int, ...], List[int]]]):
  return modules.Transpose(perm=perm)(input)


def pad(input: Tensor, pad, mode='constant', value=0) -> Tensor:
  return modules.GenericPadNd(padding=pad, mode=mode, value=value)(input)


def max(*inputs: Tensor) -> Tensor:
  return modules.Max()(*inputs)


def conv1d(
    input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
    stride: Union[int, _size] = 1, padding: Union[int, _size] = 0,
    dilation: Union[int, _size] = 1, groups: int = 1) -> Tensor:
  mod = modules.FunctionalConv1d(stride=stride, padding=padding, dilation=dilation, groups=groups)
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


def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
  return modules.LeakyReLU(negative_slope=negative_slope, inplace=inplace)(input)


def tanh(input: Tensor) -> Tensor:
  return modules.Tanh()(input)


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
