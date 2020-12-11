
from __future__ import annotations
import tensorflow as tf
import math
from typing import Optional, Dict, Any
from returnn.tf.layers.basic import LayerBase, ConvLayer
from .module import Module
from .utils import _single, _pair, _triple, _reverse_repeat_tuple, _ntuple
from ..common_types import _scalar_or_tuple_any_t, _size_1_t, _size_2_t, _size_3_t
from ...tensor import Tensor
from ..parameter import Parameter
from .. import init


class _ConvNd(Module):
  nd: Optional[int] = None  # defined by subclass
  transposed: bool = False

  def __init__(self,
               in_channels: int,
               out_channels: int,
               kernel_size: _scalar_or_tuple_any_t,
               stride: _scalar_or_tuple_any_t = 1,
               padding: _scalar_or_tuple_any_t = 0,
               dilation: _scalar_or_tuple_any_t = 1,
               transposed: Optional[bool] = None,
               output_padding: _scalar_or_tuple_any_t = 0,
               groups: int = 1,
               bias: bool = True,
               padding_mode: str = "zeros") -> None:
    super(_ConvNd, self).__init__()
    if in_channels % groups != 0:
      raise ValueError('in_channels must be divisible by groups')
    if out_channels % groups != 0:
      raise ValueError('out_channels must be divisible by groups')
    valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
    if padding_mode not in valid_padding_modes:
      raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
        valid_padding_modes, padding_mode))
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = _ntuple(self.nd)(kernel_size)
    self.stride = _ntuple(self.nd)(stride)
    self.padding = _ntuple(self.nd)(padding)
    self.dilation = _ntuple(self.nd)(dilation)
    if transposed is not None:
      self.transposed = transposed
    self.output_padding = _ntuple(self.nd)(output_padding)
    self.groups = groups
    self.padding_mode = padding_mode
    # `_reversed_padding_repeated_twice` is the padding to be passed to
    # `F.pad` if needed (e.g., for non-zero padding types that are
    # implemented as two ops: padding + conv). `F.pad` accepts paddings in
    # reverse order than the dimension.
    self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
    if self.transposed:
      self.weight = Parameter(Tensor(
        in_channels, out_channels // groups, *self.kernel_size))
    else:
      self.weight = Parameter(Tensor(
        out_channels, in_channels // groups, *self.kernel_size))
    if bias:
      self.bias = Parameter(Tensor(out_channels))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      bound = 1 / math.sqrt(fan_in)
      init.uniform_(self.bias, -bound, bound)

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    assert len(input.shape) == 2 + self.nd
    self._assert_spatial_axes_in_order(input)  # not implemented otherwise
    assert all(p == 0 for p in self.padding)  # not implemented otherwise
    assert all(p == 0 for p in self.output_padding)  # not implemented otherwise
    assert self.padding_mode == "zeros"  # not implemented otherwise
    d = {
      "class": "conv", "from": self._get_input_layer_name(input),
      "activation": None,
      "with_bias": self.bias is not None,
      "n_out": self.out_channels,
      "filter_size": self.kernel_size,
      "padding": "valid"}
    if any(s != 1 for s in self.stride):
      d["strides"] = self.stride
    if any(d != 1 for d in self.dilation):
      d["dilation_rate"] = self.dilation
    if self.groups != 1:
      d["groups"] = self.groups
    return d

  def check_returnn_layer(self, layer: ConvLayer):
    assert layer.input_data.dim == self.in_channels

  def import_params_torch_to_returnn(self, *, layer: LayerBase, torch_module: _ConvNd):
    session = tf.compat.v1.get_default_session()
    if self.transposed:
      # E.g. convert 11,13,10 -> 10,1,13,11.
      values = torch_module.weight.detach().numpy()
      assert len(self.kernel_size) in {1, 2}
      axes = list(range(values.ndim))
      values = values.transpose(*(axes[2:] + [axes[1], axes[0]]))
      if len(self.kernel_size) == 1:
        values = values[:, None]
      layer.params["W_native_transposed_conv"].load(values, session=session)
    else:
      # E.g. 384,80,7 -> 7,80,384
      values = torch_module.weight.detach().numpy()
      axes = list(range(values.ndim))
      values = values.transpose(*(axes[2:] + [axes[1], axes[0]]))
      layer.params["W"].load(values, session=session)
    if self.bias is not None:
      layer.params["bias"].load(torch_module.bias.detach().numpy(), session=session)


class Conv1d(_ConvNd):
  nd = 1

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: _size_1_t,
      stride: _size_1_t = 1,
      padding: _size_1_t = 0,
      dilation: _size_1_t = 1,
      groups: int = 1,
      bias: bool = True,
      padding_mode: str = 'zeros'
  ):
    super(Conv1d, self).__init__(
      in_channels, out_channels, kernel_size, stride,
      padding=padding, dilation=dilation,
      groups=groups, bias=bias, padding_mode=padding_mode)


class Conv2d(_ConvNd):
  nd = 2

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: _size_2_t,
      stride: _size_2_t = 1,
      padding: _size_2_t = 0,
      dilation: _size_2_t = 1,
      groups: int = 1,
      bias: bool = True,
      padding_mode: str = 'zeros'
  ):
    super(Conv2d, self).__init__(
      in_channels, out_channels, kernel_size, stride,
      padding=padding, dilation=dilation,
      groups=groups, bias=bias, padding_mode=padding_mode)


class _ConvTransposeNd(_ConvNd):
  def __init__(self, in_channels, out_channels, kernel_size, stride,
               padding, dilation, transposed, output_padding,
               groups, bias, padding_mode):
    if padding_mode != 'zeros':
      raise ValueError('Only "zeros" padding mode is supported for {}'.format(self.__class__.__name__))

    super(_ConvTransposeNd, self).__init__(
      in_channels, out_channels, kernel_size, stride,
      padding, dilation, transposed, output_padding,
      groups, bias, padding_mode)

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    assert len(input.shape) == 2 + self.nd
    self._assert_spatial_axes_in_order(input)  # not implemented otherwise
    assert self.groups == 1  # not implemented otherwise
    assert self.padding_mode == "zeros"  # not implemented otherwise
    assert all(d == 1 for d in self.dilation)
    d = {
      "class": "transposed_conv", "from": self._get_input_layer_name(input),
      "activation": None,
      "with_bias": self.bias is not None,
      "n_out": self.out_channels,
      "filter_size": self.kernel_size,
      "strides": self.stride,
      "padding": "valid",
      "output_padding": self.output_padding}
    if self.padding:
      d["remove_padding"] = self.padding
    return d


class ConvTranspose1d(_ConvTransposeNd):
  nd = 1

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: _size_1_t,
      stride: _size_1_t = 1,
      padding: _size_1_t = 0,
      output_padding: _size_1_t = 0,
      groups: int = 1,
      bias: bool = True,
      dilation: _size_1_t = 1,
      padding_mode: str = 'zeros'
  ):
    super(ConvTranspose1d, self).__init__(
      in_channels, out_channels, kernel_size, stride, padding, dilation,
      True, output_padding, groups, bias, padding_mode)


class ConvTranspose2d(_ConvTransposeNd):
  nd = 2

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: _size_2_t,
      stride: _size_2_t = 1,
      padding: _size_2_t = 0,
      output_padding: _size_2_t = 0,
      groups: int = 1,
      bias: bool = True,
      dilation: _size_2_t = 1,
      padding_mode: str = 'zeros'
  ):
    super(ConvTranspose2d, self).__init__(
      in_channels, out_channels, kernel_size, stride, padding, dilation,
      True, output_padding, groups, bias, padding_mode)


class _FunctionalConvNd(Module):
  is_original_torch_module = False
  nd: Optional[int] = None
  transposed: bool = False

  def __init__(self,
               stride: _scalar_or_tuple_any_t = 1,
               padding: _scalar_or_tuple_any_t = 0,
               dilation: _scalar_or_tuple_any_t = 1,
               output_padding: _scalar_or_tuple_any_t = 0,
               groups: int = 1,
               padding_mode: str = "zeros") -> None:
    super(_FunctionalConvNd, self).__init__()
    valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
    if padding_mode not in valid_padding_modes:
      raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
        valid_padding_modes, padding_mode))
    self.stride = _ntuple(self.nd)(stride)
    self.padding = _ntuple(self.nd)(padding)
    self.dilation = _ntuple(self.nd)(dilation)
    self.output_padding = _ntuple(self.nd)(output_padding)
    self.groups = groups
    self.padding_mode = padding_mode
    # `_reversed_padding_repeated_twice` is the padding to be passed to
    # `F.pad` if needed (e.g., for non-zero padding types that are
    # implemented as two ops: padding + conv). `F.pad` accepts paddings in
    # reverse order than the dimension.
    self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

  def create_returnn_layer_dict(self, input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Dict[str, Any]:
    assert len(input.shape) == 2 + self.nd
    self._assert_spatial_axes_in_order(input)  # not implemented otherwise
    assert len(weight.shape) == 2 + self.nd
    kernel_size = weight.shape[2:]
    out_channels, in_channels = weight.shape[:2]
    in_channels *= self.groups
    if self.transposed:
      out_channels, in_channels = in_channels, out_channels
    returnn_weight_axes = [self._get_input_axis_to_returnn(weight, i) for i in range(len(weight.shape))]
    returnn_weight_axes_ = returnn_weight_axes[2:] + [returnn_weight_axes[1], returnn_weight_axes[0]]
    returnn_weight_transpose_perm = {i: j for (i, j) in zip(returnn_weight_axes, returnn_weight_axes_)}
    if bias is not None:
      assert bias.shape == (out_channels,)
    assert self.groups == 1  # not implemented otherwise
    assert self.padding_mode == "zeros"  # not implemented otherwise
    if self.transposed:  # transposed conv
      assert all(d == 1 for d in self.dilation)
      return {
        "class": "transposed_conv", "from": self._get_input_layer_name(input),
        "n_out": out_channels,
        "activation": None,
        "with_bias": bias is not None,
        "bias": self._get_input_layer_name(bias) if bias is not None else None,
        "filter_size": kernel_size,
        "filter": self._get_input_layer_name(weight),
        "filter_perm": returnn_weight_transpose_perm,
        "padding": "valid",
        "output_padding": self.output_padding,
        "remove_padding": self.padding,
        "strides": self.stride}
    else:
      assert all(p == 0 for p in self.padding)  # not implemented otherwise
      assert all(p == 0 for p in self.output_padding)  # not implemented otherwise
      return {
        "class": "conv", "from": self._get_input_layer_name(input),
        "n_out": out_channels,
        "activation": None,
        "with_bias": bias is not None,
        "bias": self._get_input_layer_name(bias) if bias is not None else None,
        "filter_size": kernel_size,
        "filter": self._get_input_layer_name(weight),
        "filter_perm": returnn_weight_transpose_perm,
        "padding": "valid",
        "strides": self.stride,
        "dilation_rate": self.dilation}


class FunctionalConv1d(_FunctionalConvNd):
  nd = 1


class FunctionalConv2d(_FunctionalConvNd):
  nd = 2


class FunctionalConvTransposed1d(_FunctionalConvNd):
  nd = 1
  transposed = True


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
