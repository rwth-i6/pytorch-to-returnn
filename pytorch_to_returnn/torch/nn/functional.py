
from typing import Optional, Union, List, Tuple
from ..tensor import Tensor
from .._C import Size


_size = Union[Size, List[int], Tuple[int, ...]]


def pad(input: Tensor, pad, mode='constant', value=0) -> Tensor:
  return input  # TODO


def conv1d(
    input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
    stride: Union[int, _size] = 1, padding: Union[int, _size] = 0,
    dilation: Union[int, _size] = 1, groups: int = 1) -> Tensor:
  return input  # TODO


def conv_transpose1d(
    input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
    stride: Union[int, _size] = 1, padding: Union[int, _size] = 0,
    output_padding: Union[int, _size] = 0,
    groups: int = 1,
    dilation: Union[int, _size] = 1) -> Tensor:
  return input  # TODO


def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
  return input  # TODO


def tanh(input: Tensor) -> Tensor:
  return input  # TODO


def norm_except_dim(v: Tensor, pow: int = 2, dim: int = 0) -> Tensor:
  return v
