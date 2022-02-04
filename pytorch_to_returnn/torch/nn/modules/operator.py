
from typing import Optional, Tuple, Any, List, Dict, Set, Union
from returnn.tf.layers.basic import LayerBase
from returnn.tf.util.data import Dim
from .module import Module
from .shape import SizeValue
from ...tensor import Tensor, dtype as _dtype
from ....naming import Naming, TensorEntry


class Copy(Module):
  is_original_torch_module = False

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    return {"class": "copy", "from": self._get_input_layer_name(input)}

  def make_output_tensor_from_returnn(self, inputs_flat: List[Tensor], layer: LayerBase) -> Tensor:
    assert len(inputs_flat) == 1
    return inputs_flat[0]


class Range(Module):
  is_original_torch_module = False

  def create_returnn_layer_dict(self, limit, start, delta, dtype, sparse=False) -> Dict[str, Any]:
    if isinstance(limit, Tensor):
      return {"class": "range_from_length", "from": self._get_input_layer_name(limit)}
    else:
      assert isinstance(limit, int)
      return {"class": "range", "limit": limit, "start": start, "delta": delta, "dtype": dtype, "sparse": sparse}

  def _get_output_shape_from_returnn(self,
                                     inputs_flat: List[Optional[Union[Tensor, int, bool]]], layer: LayerBase
                                     ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    from .shape import SizeValue
    limit, start, delta, *_ = inputs_flat
    size = None
    if isinstance(limit, Tensor):
      assert limit.is_defined
      limit_size = limit.returnn_naming_entry.is_size_value
      if limit_size is not None:
        size = (limit_size - int(start)) // int(delta)
      else:
        limit = limit.numpy()
    if size is None:
      size = SizeValue((int(limit) - int(start)) // int(delta))
    torch_shape = (size,)
    returnn_axis_from_torch_axis = {0: 0}
    return torch_shape, returnn_axis_from_torch_axis


class RandInt(Module):
  is_original_torch_module = False
  is_deterministic = False

  def create_returnn_layer_dict(self, low, high, size, dtype=None) -> Dict[str, Any]:
    dtype = dtype or "int64"
    if isinstance(low, Tensor):
      low = low.type(dtype)
      low = self._get_input_layer_name(low)
    if isinstance(high, Tensor):
      high = high.type(dtype)
      high = self._get_input_layer_name(high)
    naming = Naming.get_instance()
    call = naming.module_call_stack[-1]
    assert call.module.module is self
    source = []
    for sz in size:
      if isinstance(sz, Tensor):
        tensor_entry = naming.tensors[sz]
        assert tensor_entry.is_size_value is not None
        for originating_tensor in tensor_entry.is_size_value.get_originating_tensors():
          if naming.tensors[originating_tensor] not in call.inputs_tensor_deps:
            source.append(self._get_input_layer_name(originating_tensor))
            # add dependency to get complete in feed_dict in Module.make_output_tensor_from_returnn
            call.inputs_tensor_deps.append(naming.tensors[originating_tensor])
    size = tuple(naming.tensors[sz].is_size_value.dim_tag if isinstance(sz, Tensor) else sz for sz in size)
    assert not None in size
    d = {"class": "rand_int", "shape": size, "maxval": high, "minval": low, "dtype": dtype}
    if source:
      d["from"] = list(source)
    return d

  def _get_output_shape_from_returnn(self,
                                     inputs_flat: List[Optional[Union[Tensor, int, bool]]], layer: LayerBase
                                     ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    _, _, *size, _ = inputs_flat

    torch_shape = tuple(self._to_size_value(sz) for sz in size)
    returnn_axis_from_torch_axis = {i: i for i in range(len(torch_shape))}
    return torch_shape, returnn_axis_from_torch_axis

  def get_returnn_name(self) -> str:
    # Used to allow finding this module in the namespace
    return "randint"

  @staticmethod
  def _to_size_value(x: Union[int, Tensor]) -> SizeValue:
    x_size = None
    if isinstance(x, Tensor):
      assert x.is_defined
      x_size = x.returnn_naming_entry.is_size_value
      x = x.numpy()
    sv = SizeValue(int(x))
    if x_size is not None:
      sv.dim_tag = x_size.dim_tag
      sv.originating_tensor = x_size.originating_tensor
    return sv


class Cat(Module):
  is_original_torch_module = False

  def __init__(self, dim=0):
    super(Cat, self).__init__()
    self.dim = dim

  def create_returnn_layer_dict(self, *inputs: Tensor) -> Dict[str, Any]:
    assert len(inputs) > 0
    inputs, _ = _unify_tensor_axes_returnn_meta(*inputs, concat_axes=[self.dim])
    cat_axis = self.dim
    if cat_axis < 0:
      cat_axis += len(inputs[0].shape)
    assert 0 <= cat_axis <= len(inputs[0].shape)
    sources = []
    for input in inputs:
      assert len(inputs[0].shape) == len(input.shape)
      assert all(d == d0 for i, (d, d0) in enumerate(zip(input.shape, inputs[0].shape)) if i != cat_axis)
      returnn_axis = self._get_input_axis_to_returnn(input, axis=cat_axis)
      sources.append((self._get_input_layer_name(input), returnn_axis))
    return {"class": "concat", "from": sources}

  def _get_output_shape_from_returnn(self,
                                     inputs_flat: List[Optional[Union[Tensor, int, bool]]], layer: LayerBase
                                     ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    assert len(inputs_flat) > 0
    cat_axis = self.dim
    if cat_axis < 0:
      cat_axis += len(inputs_flat[0].shape)
    assert 0 <= cat_axis <= len(inputs_flat[0].shape)
    torch_shape = list(inputs_flat[0].shape)
    for input_ in inputs_flat[1:]:
      assert input_.ndim == len(torch_shape)
      torch_shape[cat_axis] += input_.shape[cat_axis]
    _, returnn_axis_from_torch_axis = super(Cat, self)._get_output_shape_from_returnn([inputs_flat[0]], layer=layer)
    return tuple(torch_shape), returnn_axis_from_torch_axis


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
    inputs, out_shape = _unify_tensor_axes_returnn_meta(*inputs)
    return {
      "class": "combine", "kind": self.kind, "out_shape": out_shape,
      "from": [self._get_input_layer_name(input) for input in inputs]}


class ComparisonOperator(BinaryOperator):
  is_original_torch_module = False

  def __init__(self, kind: str):
    """
    :param str kind: e.g. "equal", "greater", "less" or other supported TF comparison ops
    """
    super(ComparisonOperator, self).__init__(kind=kind)

  def create_returnn_layer_dict(self, *inputs: Tensor):
    inputs, out_shape = _unify_tensor_axes_returnn_meta(*inputs)
    return {
      "class": "compare", "kind": self.kind, "out_shape": out_shape,
      "from": [self._get_input_layer_name(input) for input in inputs]}


class Minimum(Module):
  is_original_torch_module = False

  def create_returnn_layer_dict(self, *inputs: Tensor):
    inputs, out_shape = _unify_tensor_axes_returnn_meta(*inputs)
    return {
      "class": "eval", "eval": "tf.minimum(source(0), source(1))",
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


class ReturnnReinterpretSameSizeAs(Module):
  """
  Used by :func:`_unify_tensor_axes_returnn_meta`.
  """
  is_original_torch_module = False

  def create_returnn_layer_dict(self, input: Tensor, same_as: Tensor) -> Dict[str, Any]:
    return {
      "class": "reinterpret_data",
      "from": self._get_input_layer_name(input),
      "size_base": self._get_input_layer_name(same_as)}

  def _get_output_shape_from_returnn(self,
                                     inputs_flat: List[Tensor], layer: LayerBase
                                     ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    input, _ = inputs_flat
    assert isinstance(input, Tensor)
    naming = Naming.get_instance()
    x = naming.tensors[input]
    assert isinstance(x, TensorEntry)
    return input.shape, x.returnn_axis_from_torch_axis  # no change


class ReturnnReinterpretSetAxes(Module):
  """
  Used by :func:`_unify_tensor_axes_returnn_meta`.
  """
  is_original_torch_module = False

  def __init__(self, *,
               dims_by_key: Optional[Dict[str, int]] = None,  # "B"|"T"|"F" are the keys
               batch_dim: Optional[int] = None,
               time_dim: Optional[int] = None,
               feature_dim: Optional[int] = None):
    super(ReturnnReinterpretSetAxes, self).__init__()
    self.dims_by_key = dims_by_key
    self.batch_dim = batch_dim
    self.time_dim = time_dim
    self.feature_dim = feature_dim

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    axes = {}
    if self.dims_by_key:
      for key, dim in self.dims_by_key.items():
        axes[key] = self._get_input_axis_to_returnn(input, dim)
    for key, dim in [("B", self.batch_dim), ("T", self.time_dim), ("F", self.feature_dim)]:
      if dim is None:
        continue
      axes[key] = self._get_input_axis_to_returnn(input, dim)
    return {
      "class": "reinterpret_data",
      "from": self._get_input_layer_name(input),
      "set_axes": axes}

  def _get_output_shape_from_returnn(self,
                                     inputs_flat: List[Tensor], layer: LayerBase
                                     ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    input, = inputs_flat
    assert isinstance(input, Tensor)
    naming = Naming.get_instance()
    x = naming.tensors[input]
    assert isinstance(x, TensorEntry)
    return input.shape, x.returnn_axis_from_torch_axis  # no change


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

    inv_perm = {j: i for i, j in perm.items()}
    out_torch_shape = [input.shape[perm[i]] for i in range(ndim)]
    out_returnn_axis_from_torch_axis = {
      inv_perm[i]: j for (i, j) in tensor_entry.returnn_axis_from_torch_axis.items()}
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


class GatherTensor(Module):
  """
  Basically x[pos] but in specific dim (axis) where pos is a tensor.
  """
  is_original_torch_module = False

  def __init__(self, dim: int):
    super(GatherTensor, self).__init__()
    self.dim = dim

  def create_returnn_layer_dict(self, input: Tensor, pos: Tensor) -> Dict[str, Any]:
    return {
      "class": "gather", "from": self._get_input_layer_name(input),
      "axis": self._get_input_axis_to_returnn(input, axis=self.dim),
      "position": self._get_input_layer_name(pos)}

  def _get_output_shape_from_returnn(self,
                                     inputs_flat: List[Tensor], layer: LayerBase
                                     ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    input, pos = inputs_flat
    _, returnn_axis_from_torch_axis = super(GatherTensor, self)._get_output_shape_from_returnn(inputs_flat, layer)
    out_shape = list(input.shape)
    out_shape[self.dim:self.dim + 1] = pos.shape
    return tuple(out_shape), returnn_axis_from_torch_axis


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

  def _get_output_shape_from_returnn(self, inputs_flat: List[Tensor], layer: LayerBase
                                     ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    """
    The size of the dynamic axes might be changed, so we have to take care of this here for the torch shape.
    """
    torch_shape, returnn_axis_from_torch_axis = super(Slice, self)._get_output_shape_from_returnn(
      inputs_flat=inputs_flat, layer=layer)
    assert len(inputs_flat) == 1
    torch_shape = list(inputs_flat[0].shape)
    start = self.start or 0
    stop = self.stop or torch_shape[self.axis]
    step = self.step or 1
    if start < 0:
      start += torch_shape[self.axis]
    if stop < 0:
      stop += torch_shape[self.axis]
    assert 0 <= start <= torch_shape[self.axis] - 1
    assert 0 <= stop <= torch_shape[self.axis]
    torch_shape[self.axis] = len(range(start, stop, step))
    return tuple(torch_shape), returnn_axis_from_torch_axis


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


class Reduce(Module):
  """
  Wraps RETURNN ReduceLayer.
  """
  is_original_torch_module = False

  def __init__(self, mode: str, axes: Optional[Union[List[int], int]]):
    assert mode in ["sum", "max", "argmin", "min", "argmax", "mean", "logsumexp"]
    super(Reduce, self).__init__()
    self.mode = mode
    self.axes = axes

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    axes = self.axes
    if axes is None:
      axes = range(input.ndim)
    if isinstance(axes, int):
      axes = [axes]
    axes = [self._get_input_axis_to_returnn(input, axis) for axis in axes]
    return {"class": "reduce", "mode": self.mode, "axes": axes, "from": self._get_input_layer_name(input)}


class Length(Module):
  """
  Wraps RETURNN LengthLayer.
  """
  is_original_torch_module = False

  def __init__(self, axis: int):
    super(Length, self).__init__()
    self.axis = axis

  def create_returnn_layer_dict(self, input: Tensor) -> Dict[str, Any]:
    return {
      "class": "length", "axis": self._get_input_axis_to_returnn(input, self.axis),
      "from": self._get_input_layer_name(input)}


def _unify_tensor_axes_returnn_meta(
    *inputs: Tensor, concat_axes: Optional[List[int]] = None) -> Tuple[Tuple[Tensor, ...], Set[Dim]]:
  """
  This is called when the inputs are supposed to be potentially broadcasted against each other.

  You have multiple inputs which can potentially have different dynamic axes (see RETURNN :class:`Data`),
  and this would add ``reinterpret_data`` layers when needed
  to make sure the seq lengths / dim tags are the same.

  It also removes any broadcasting dims because RETURNN will handle broadcasting automatically
  but only reliable when the dim is not present.
  """
  assert len(inputs) >= 1
  naming = Naming.get_instance()
  tensors = [naming.tensors[x] for x in inputs if isinstance(x, Tensor)]  # type: List[TensorEntry]
  if len(inputs) == 1:
    return inputs, set(tensors[0].returnn_data.dim_tags) if tensors else set()
  tensors = [x for x in tensors if x.returnn_data.batch_ndim > 0]  # filter out scalars
  if len(tensors) <= 1:
    return inputs, set(tensors[0].returnn_data.dim_tags) if tensors else set()
  inputs = list(inputs)

  # Collect broadcast dims and out_shape.
  # We will squeeze them out later.
  max_ndim = max(x.returnn_data.batch_ndim for x in tensors)
  concat_axes = [ax + max_ndim if ax < 0 else ax for ax in concat_axes or []]
  broadcast_axes = [set() for _ in inputs]  # input idx -> set of (negative) broadcast Torch axes
  out_shape = []
  for torch_axis in range(max_ndim):
    neg_torch_axis = torch_axis - max_ndim
    dims_for_axis = []  # type: List[Union[None, Dim]]  # None -> not existing, dim 1 -> maybe broadcast
    for x in inputs:
      if not isinstance(x, Tensor):
        dims_for_axis.append(None)
        continue
      x = naming.tensors[x]
      assert isinstance(x, TensorEntry)
      if x.returnn_data.batch_ndim < abs(neg_torch_axis):
        dims_for_axis.append(None)
      else:
        torch_axis = x.returnn_data.batch_ndim + neg_torch_axis
        assert 0 <= torch_axis < x.returnn_data.batch_ndim
        returnn_axis = x.returnn_axis_from_torch_axis[torch_axis]
        dims_for_axis.append(x.returnn_data.dim_tags[returnn_axis])
    assert len(dims_for_axis) == len(inputs)

    broadcast_inputs_for_axis = set()  # input indices
    dim_for_axis = None
    if torch_axis in concat_axes:
      dim_for_axis = sum(dims_for_axis)
    else:
      for i, (x, dim) in enumerate(zip(inputs, dims_for_axis)):
        if dim is None:
          continue
        assert isinstance(dim, Dim)
        if dim.dimension == 1 and any(d for d in dims_for_axis if d is not None and d.dimension != 1):
          broadcast_inputs_for_axis.add(i)
          continue
        assert all(dim.dimension == d.dimension for d in dims_for_axis if d is not None and d.dimension != 1), (
          f"invalid input {x} axis {i} dim {dim}")
        if not dim_for_axis:
          dim_for_axis = dim
    assert dim_for_axis
    out_shape.append(dim_for_axis)
    for idx in broadcast_inputs_for_axis:
      broadcast_axes[idx].add(neg_torch_axis)
  assert len(set(out_shape)) == len(out_shape) == max_ndim

  # Potentially reset dynamic dim tags to reflect same dim.
  dims = {}  # torch axis (negative) -> (TensorEntry, DimensionTag) (static != 1, or dynamic)
  for i, x in enumerate(inputs):
    if not isinstance(x, Tensor):
      continue
    x = naming.tensors[x]
    assert isinstance(x, TensorEntry)
    if x.returnn_data.batch_ndim == 0:  # scalars are always fine
      continue
    used_reinterpret_same_size = False
    for axis in range(-1, -x.returnn_data.batch_ndim - 1, -1):
      returnn_axis = x.returnn_axis_from_torch_axis[axis + x.returnn_data.batch_ndim]
      dim_tag = x.returnn_data.get_dim_tag(returnn_axis)
      if dim_tag.dimension == 1:
        continue  # broadcast dim, so always ok. do not add
      if axis + x.returnn_data.batch_ndim in concat_axes:
        continue  # concat dim, so do not add
      if axis not in dims:
        dims[axis] = (x, dim_tag)
      else:
        prev_x, prev_dim_tag = dims[axis]
        if prev_dim_tag == dim_tag:
          pass  # ok
        elif prev_dim_tag.dimension is not None:
          # Static dimension, need to match.
          assert dim_tag.dimension == prev_dim_tag.dimension, f"not matching Data {x.returnn_data}"
        else:
          assert not used_reinterpret_same_size  # It should not happen to use this multiple times.
          # This is the case we wanted to catch here.
          # We have two different dynamic axes, which are expected to represent the same seq lengths.
          # So we unify them via reinterpret_data.
          x_ = ReturnnReinterpretSameSizeAs()(x.tensor(), prev_x.tensor())
          x = naming.tensors[x_]
          assert isinstance(x, TensorEntry)
          inputs[i] = x_
          used_reinterpret_same_size = True
  del dims

  # Squeeze out all the broadcast dims.
  from .shape import Squeeze
  for i, x in enumerate(inputs):
    broadcast_dims_for_input = sorted(broadcast_axes[i])
    if broadcast_dims_for_input:
      inputs[i] = Squeeze(dim=broadcast_dims_for_input)(x)

  return tuple(inputs), set(out_shape)


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
