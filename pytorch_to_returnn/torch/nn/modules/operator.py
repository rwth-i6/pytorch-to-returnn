
from typing import Optional, Tuple, Any, List, Dict, Union
from returnn.tf.layers.basic import LayerBase
from .module import Module
from ...tensor import Tensor, dtype as _dtype
from ....naming import Naming, TensorEntry


class Copy(Module):
  is_original_torch_module = False

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    return {"class": "copy", "from": self._get_input_layer_name(input)}

  def make_output_tensor_from_returnn(self, inputs_flat: List[Tensor], layer: LayerBase) -> Tensor:
    assert len(inputs_flat) == 1
    return inputs_flat[0]


class GetSublayer(Module):
  is_original_torch_module = False

  def __init__(self, sub_layer: str):
    super(GetSublayer, self).__init__()
    self.sub_layer = sub_layer

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    return {"class": "copy", "from": f"{self._get_input_layer_name(input)}/{self.sub_layer}"}


class Cast(Module):
  is_original_torch_module = False

  def __init__(self, dtype: Union[str, _dtype]):
    super(Cast, self).__init__()
    self.dtype = _dtype(dtype)

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    return {"class": "cast", "from": self._get_input_layer_name(input), "dtype": self.dtype.name}


class BinaryOperator(Module):
  is_original_torch_module = False

  def __init__(self, kind: str):
    """
    :param kind: "add", "sub", "mul", "truediv"
    """
    super(BinaryOperator, self).__init__()
    self.kind = kind

  def get_returnn_name(self) -> str:
    return self.kind

  def create_returnn_layer_dict(self, *inputs: Tensor):
    inputs = _unify_tensor_dyn_axes(*inputs)
    return {
      "class": "combine", "kind": self.kind,
      "from": [self._get_input_layer_name(input) for input in inputs]}


class ComparisonOperator(BinaryOperator):
  is_original_torch_module = False

  def __init__(self, kind: str):
    """
    :param str kind: e.g. "equal", "greater", "less" or other supported TF comparison ops
    """
    super(ComparisonOperator, self).__init__(kind=kind)

  def create_returnn_layer_dict(self, *inputs: Tensor):
    inputs = _unify_tensor_dyn_axes(*inputs)
    return {
      "class": "compare", "kind": self.kind,
      "from": [self._get_input_layer_name(input) for input in inputs]}


class Reciprocal(Module):
  """
  1/x or 1/max(eps,x)
  """
  is_original_torch_module = False

  def __init__(self, eps: Optional[float] = None):
    super(Reciprocal, self).__init__()
    self.eps = eps

  def create_returnn_layer_dict(self, input: Tensor):
    x = "source(0)"
    if self.eps is not None:
      x = f"maximum_with_identity_grad({x})"
    return {
      "class": "eval", "eval": f"tf_compat.v1.reciprocal({x})",
      "from": self._get_input_layer_name(input)}


class Max(Module):
  is_original_torch_module = False
  pass  # TODO


class ReturnnReinterpretSameSizeAs(Module):
  """
  Used by :func:`_unify_tensor_dyn_axes`.
  """
  is_original_torch_module = False

  def create_returnn_layer_dict(self, input: Tensor, same_as: Tensor) -> Dict[str, Any]:
    return {
      "class": "reinterpret_data",
      "from": self._get_input_layer_name(input),
      "size_base": self._get_input_layer_name(same_as)}


class Transpose(Module):
  """
  Note: The resulting Torch tensor is transposed as expected.
  However, on the RETURNN side, we actually should never need to transpose,
  as we have dimension tags, and all layers should refer to axes by dim tags.
  So on RETURNN side, this is a no-op.
  """
  is_original_torch_module = False

  def __init__(self, perm: Optional[Union[Dict[int, int], Tuple[int, ...], List[int]]]):
    super(Transpose, self).__init__()
    if perm is None:
      self.perm = None
    else:
      if isinstance(perm, (list, tuple)):
        perm = dict(enumerate(perm))
      assert isinstance(perm, dict)
      self.perm = perm  # type: Optional[Dict[int, int]]
      assert len(perm) == len(set(perm.values()))  # should be unique

  def _get_perm(self, input: Tensor) -> Dict[int, int]:
    if self.perm is None:
      dims = range(len(input.shape))
      return {i: j for (i, j) in zip(dims, reversed(dims))}

    def _axis(i):
      assert -len(input.shape) <= i < len(input.shape)
      if i < 0:
        i += len(input.shape)
      assert 0 <= i < len(input.shape)
      return i
    perm = {_axis(i): _axis(j) for (i, j) in self.perm.items()}
    assert len(perm) == len(set(perm.values()))  # should be unique
    return perm

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    # See comment in class docstring. No need to actually transpose.
    return {"class": "copy", "from": self._get_input_layer_name(input)}

  def _get_output_shape_from_returnn(self,
                                     inputs_flat: List[Tensor], layer: LayerBase
                                     ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    """
    :return: (torch_shape, returnn_axis_from_torch_axis).
      Torch shape how it would have looked when this would be processed within Torch.
      The RETURNN layer.output shape (order of axes) might look different.

    On RETURNN side, this is a no-op.
    But on Torch side, we transpose as expected.
    """
    input, = inputs_flat
    perm = self._get_perm(input)
    target_axes = set(perm)
    source_axes = set(perm.values())
    ndim = len(input.shape)
    rem_target_axes = [i for i in range(ndim) if i not in target_axes]
    rem_source_axes = [i for i in range(ndim) if i not in source_axes]
    assert len(rem_target_axes) == len(rem_source_axes)
    perm.update({i: j for (i, j) in zip(rem_target_axes, rem_source_axes)})
    assert len(perm) == len(set(perm.values())) == ndim

    naming = Naming.get_instance()
    tensor_entry = naming.tensors[input]
    assert isinstance(tensor_entry, TensorEntry)
    assert tensor_entry.returnn_data and tensor_entry.returnn_axis_from_torch_axis
    assert tensor_entry.returnn_data.batch_ndim == ndim

    out_torch_shape = [input.shape[perm[i]] for i in range(ndim)]
    out_returnn_axis_from_torch_axis = {
      perm[i]: j for (i, j) in tensor_entry.returnn_axis_from_torch_axis.items()}
    return tuple(out_torch_shape), out_returnn_axis_from_torch_axis


class Gather(Module):
  """
  Basically x[pos] but in specific dim (axis).
  """
  is_original_torch_module = False

  def __init__(self, dim: int, pos: int):
    super(Gather, self).__init__()
    self.dim = dim
    self.pos = pos

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    return {
      "class": "gather", "from": self._get_input_layer_name(input),
      "axis": self._get_input_axis_to_returnn(input, axis=self.dim),
      "position": self.pos}


class Slice(Module):
  """
  Slicing of tensors. Wraps RETURNN's SliceLayer.
  """
  is_original_torch_module = False

  def __init__(self, axis: int, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None):
    super(Slice, self).__init__()
    self.axis = axis
    self.start = start
    self.stop = stop
    self.step = step

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    return {
      "class": "slice", "axis": self._get_input_axis_to_returnn(input, axis=self.axis),
      "slice_start": self.start, "slice_end": self.stop, "slice_step": self.step,
      "from": self._get_input_layer_name(input)}


class Stack(Module):
  """
  Wraps RETURNN StackLayer.
  """
  is_original_torch_module = False

  def __init__(self, dim: Optional[int] = None):
    super(Stack, self).__init__()
    self.dim = dim

  def create_returnn_layer_dict(self, *inputs: Tensor) -> Dict[str, Any]:
    return {
      "class": "stack", "axis": self.dim,
      "from": [self._get_input_layer_name(x) for x in inputs]}


class Tile(Module):
  """
  Wraps RETURNN TileLayer.
  """
  is_original_torch_module = False

  def __init__(self, multiples: Dict[int, int]):
    super(Tile, self).__init__()
    assert isinstance(multiples, dict)
    self.multiples = multiples

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    multiples = {
      self._get_input_axis_to_returnn(input, axis=axis): multiple
      for axis, multiple in self.multiples.items()}
    return {
      "class": "tile", "multiples": multiples,
      "from": self._get_input_layer_name(input)}


def _unify_tensor_dyn_axes(*inputs: Tensor) -> Tuple[Tensor, ...]:
  """
  You have multiple inputs which can potentially have different dynamic axes (see RETURNN :class:`Data`),
  and this would add ``reinterpret_data`` layers when needed
  to make sure the seq lengths / dim tags are the same.
  """
  if len(inputs) <= 1:
    return inputs
  naming = Naming.get_instance()
  tensors = [naming.tensors[x] for x in inputs]  # type: List[TensorEntry]
  x0 = tensors[0]
  num_dims = max(x.returnn_data.batch_ndim for x in tensors)
  assert all(x.returnn_data.batch_ndim in {0, num_dims} for x in tensors)
  num_spatial_dims = len(x0.returnn_data.get_spatial_batch_axes())
  if num_spatial_dims == 0:
    return inputs
  # Assume same order of spatial axes, but not matter where B/F is.
  spatial_dims = {}  # spatial idx -> (tensor, DimensionTag) (static != 1, or dynamic)
  for i, x in enumerate(list(tensors)):
    if x.returnn_data.batch_ndim == 0:  # scalars are fine
      continue
    x_spatial_axes = x.returnn_data.get_spatial_batch_axes()
    assert len(x_spatial_axes) == num_spatial_dims, f"unexpected Data {x.returnn_data}"
    used_reinterpret = False
    for spatial_idx in range(num_spatial_dims):
      dim_tag = x.returnn_data.get_dim_tag(x_spatial_axes[spatial_idx])
      if dim_tag.dimension == 1:
        continue  # broadcast dim, so always ok. do not add
      if spatial_idx not in spatial_dims:
        spatial_dims[spatial_idx] = (x, dim_tag)
      else:
        prev_x, prev_dim_tag = spatial_dims[spatial_idx]
        if prev_dim_tag is dim_tag:
          pass  # ok
        elif prev_dim_tag.dimension is not None:
          # Static dimension, need to match.
          assert dim_tag.dimension == prev_dim_tag.dimension, f"not matching Data {x.returnn_data}"
        else:
          assert not used_reinterpret  # It should not happen to use this multiple times.
          # This is the case we wanted to catch here.
          # We have two different dynamic axes, which are expected to represent the same seq lengths.
          # So we unify them via reinterpret_data.
          x_ = ReturnnReinterpretSameSizeAs()(x.tensor(), prev_x.tensor())
          x = naming.tensors[x_]
          tensors[i] = x
          used_reinterpret = True
  return tuple([x.tensor() for x in tensors])


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
