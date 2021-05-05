
from ...tensor import Tensor
from .module import Module
from returnn.tf.layers.basic import LayerBase
from typing import Union, Tuple, Optional, List, Dict
from .utils import _pair, _quadruple, _ntuple
from ..common_types import _size_2_t, _size_4_t, _size_6_t
from ....naming import Naming


class GenericPadNd(Module):
  padding: Tuple[int, ...] = None  # set by instance
  mode: str = None  # set by subclass
  value: float = None  # set by subclass instance
  nd: int = None  # set by subclass

  def __init__(self, *, padding: Tuple[int, ...], mode: Optional[str] = None, value: Optional[float] = None):
    super(GenericPadNd, self).__init__()
    self.padding = padding
    if mode is not None:
      self.mode = mode
    if value is not None:
      self.value = value

  def create_returnn_layer_dict(self, input: Tensor):
    assert self.mode
    assert self.mode != "replicate"  # not implemented
    assert len(self.padding) % 2 == 0, 'Padding needs to be a multiple of 2'

    naming = Naming.get_instance()
    input_naming = naming.tensors[input]

    # These are the axes in which RETURNN will pad.
    spatial_returnn = input_naming.returnn_data.get_axes_from_description("spatial")
    # This stores the RETURNN axes which map to the torch axes we want to pad
    spatial_torch_mapped = [input_naming.returnn_axis_from_torch_axis[i]
                            for i in range(len(input.shape) - len(self.padding) // 2, len(input.shape))]

    # PyTorch specifies the padding in one big tuple
    # e.g. (1, 1, 2, 2) to pad by 2 in the last and by 1 in the second to last axis in each direction.
    # RETURNN however takes padding split by axis
    # e.g. [(1,1), (2,2)].
    # This is not yet in the correct order of the spatial axes.
    split_padding = [(self.padding[2 * i], self.padding[2 * i + 1])
                     for i in range((len(self.padding) // 2) - 1, -1, -1)]

    returnn_padding = [(0, 0) for _ in range(len(spatial_returnn))]
    for i, padding in enumerate(split_padding):
      index = spatial_returnn.index(spatial_torch_mapped[i])
      returnn_padding[index] = padding

    # PyTorch assumes the input to be in batch-feature-major.
    # E.g. for 1D, it assumes input (N, C, W_in),
    # and produces output (N, C, W_out) with W_out = W_in + padding_left + padding_right.
    # For 2D, it assumes input (N, C, H_in, W_in).
    # For 3D, it assumes input (N, C, D_in, H_in, W_in).
    # I.e. does padding in the spatial axes.
    d = {
      "class": "pad", "mode": self.mode, "axes": "spatial", "padding": returnn_padding,
      "from": self._get_input_layer_name(input)}
    if self.mode == "constant":
      d["value"] = self.value
    return d

  def _get_output_shape_from_returnn(self, inputs_flat: List[Tensor], layer: LayerBase
                                     ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    """
    The size of the dynamic axes might be changed, so we have to take care of this here for the torch shape.
    """
    torch_shape, returnn_axis_from_torch_axis = super(GenericPadNd, self)._get_output_shape_from_returnn(
      inputs_flat=inputs_flat, layer=layer)
    assert len(inputs_flat) == 1
    torch_shape = list(inputs_flat[0].shape)
    for idx in range(len(self.padding) // 2):
      torch_shape[-1 - idx] += self.padding[2 * idx] + self.padding[2 * idx + 1]
    return tuple(torch_shape), returnn_axis_from_torch_axis


class _ConstantPadNd(GenericPadNd):
  mode = "constant"

  def __init__(self, *, padding: Tuple[int, ...], value: float) -> None:
    super(_ConstantPadNd, self).__init__(padding=padding)
    self.value = value


class ConstantPad1d(_ConstantPadNd):
  nd = 1

  def __init__(self, padding: _size_2_t, value: float):
    super(ConstantPad1d, self).__init__(padding=_pair(padding), value=value)


class _ReflectionPadNd(GenericPadNd):
  mode = "reflect"


class ReflectionPad1d(_ReflectionPadNd):
  nd = 1

  def __init__(self, padding: _size_2_t):
    super(ReflectionPad1d, self).__init__(padding=_pair(padding))


class _ReplicationPadNd(GenericPadNd):
  mode = "replicate"


class ReplicationPad1d(_ReplicationPadNd):
  nd = 1

  def __init__(self, padding: _size_2_t) -> None:
    super(ReplicationPad1d, self).__init__(padding=_pair(padding))


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
