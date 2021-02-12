
from typing import Optional, Tuple, Any, List, Dict, Union
from collections import OrderedDict
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

class Cat(Module):
  is_original_torch_module = False

  def __init__(self, dim=0):
    super(Cat, self).__init__()
    self.dim = dim

  def create_returnn_layer_dict(self, *inputs: Tensor) -> Dict[str, Any]:
    naming = Naming.get_instance()
    for input in inputs:
      dim = self.dim
      assert -len(input.shape) <= dim < len(input.shape)
      if dim < 0:
        dim += len(input.shape)
      assert 0 <= dim < len(input.shape)
      input_naming = naming.tensors[input]
      returnn_axis = input_naming.returnn_axis_from_torch_axis[dim]
      assert returnn_axis == input_naming.returnn_data.feature_dim_axis, "Concatenation in dimensions other than the feature dimension is currently not supported."
    return {"class": "copy", "from": [self._get_input_layer_name(input) for input in inputs]}

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
    inputs = _unify_tensor_axes_returnn_meta(*inputs)
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
    inputs = _unify_tensor_axes_returnn_meta(*inputs)
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


def _unify_tensor_axes_returnn_meta(*inputs: Tensor) -> Tuple[Tensor, ...]:
  """
  You have multiple inputs which can potentially have different dynamic axes (see RETURNN :class:`Data`),
  and this would add ``reinterpret_data`` layers when needed
  to make sure the seq lengths / dim tags are the same.

  It also makes sure that feature_dim_axis and time_dim_axis (the RETURNN flags) are set consistently.

  It also makes sure that broadcasting in the batch-dim axis works correctly
  by potentially removing such a batch broadcast dim.
  """
  if len(inputs) <= 1:
    return inputs
  naming = Naming.get_instance()
  tensors = [naming.tensors[x] for x in inputs if isinstance(x, Tensor)]  # type: List[TensorEntry]
  tensors = [x for x in tensors if x.returnn_data.batch_ndim > 0]  # filter out scalars
  if len(tensors) <= 1:
    return inputs
  inputs = list(inputs)
  num_dims = max(x.returnn_data.batch_ndim for x in tensors)
  for idx in range(len(inputs)):
    if naming.tensors[inputs[idx]] in tensors and inputs[idx].ndim < num_dims:
      inputs[idx] = inputs[idx].expand([1] * (num_dims - inputs[idx].ndim) + [-1] * inputs[idx].ndim)

  dims = {}  # torch axis -> (tensor, DimensionTag) (static != 1, or dynamic)
  for i, x in enumerate(inputs):
    if not isinstance(x, Tensor):
      continue
    x = naming.tensors[x]
    assert isinstance(x, TensorEntry)
    if x.returnn_data.batch_ndim == 0:  # scalars are fine
      continue
    assert x.returnn_data.batch_ndim == num_dims
    used_reinterpret_same_size = False
    for axis in range(num_dims):
      returnn_axis = x.returnn_axis_from_torch_axis[axis]
      dim_tag = x.returnn_data.get_dim_tag(returnn_axis)
      if dim_tag.dimension == 1:
        continue  # broadcast dim, so always ok. do not add
      if axis not in dims:
        dims[axis] = (x, dim_tag)
      else:
        prev_x, prev_dim_tag = dims[axis]
        if prev_dim_tag is dim_tag:
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
          assert x.returnn_data.batch_ndim == num_dims
          inputs[i] = x_
          used_reinterpret_same_size = True

  # Figure out in which axis we have the batch/time/feature axis.
  special_axes_names = [("B", "batch_dim_axis"), ("T", "time_dim_axis"), ("F", "feature_dim_axis")]
  torch_special_axes = {}  # "B"|"T"|"F" -> axis->dict
  for i, x in enumerate(inputs):
    if not isinstance(x, Tensor):
      continue
    x = naming.tensors[x]
    assert isinstance(x, TensorEntry)
    for key, axis_name in special_axes_names:
      returnn_axis = getattr(x.returnn_data, axis_name)
      if returnn_axis is None:
        continue
      returnn_dim = x.returnn_data.batch_shape[returnn_axis]
      torch_axis = x.torch_axis_from_returnn_axis[returnn_axis]
      if key not in torch_special_axes:
        torch_special_axes[key] = OrderedDict({torch_axis: returnn_dim})
      elif torch_axis not in torch_special_axes[key] or torch_special_axes[key][torch_axis] == 1:
        assert key != "B", f"mismatch in batch axis in {inputs!r}"
        torch_special_axes[key][torch_axis] = returnn_dim

  axes_remap_per_input = {}  # input idx -> remap dict
  for key, axis_name in special_axes_names:
    if key not in torch_special_axes:
      continue
    torch_axes = torch_special_axes[key]
    if len(torch_axes) <= 1:
      continue
    assert key != "B"  # should be caught already...
    if any(d != 1 for _, d in torch_axes.items()):
      torch_axes = {axis: dim for (axis, dim) in torch_axes.items() if dim != 1}  # filter out broadcast specs
    torch_ref_axis, torch_ref_dim = list(torch_axes.items())[0]
    assert all(torch_ref_dim == d for _, d in torch_axes.items())
    for i, x in enumerate(inputs):
      if not isinstance(x, Tensor):
        continue
      x = naming.tensors[x]
      assert isinstance(x, TensorEntry)
      if x.returnn_data.batch_ndim == 0:  # ignore scalars
        continue
      returnn_axis = getattr(x.returnn_data, axis_name)
      if returnn_axis is None or x.torch_axis_from_returnn_axis[returnn_axis] != torch_ref_axis:
        axes_remap_per_input.setdefault(i, {})[key] = torch_ref_axis
  for i, axes_remap in axes_remap_per_input.items():
    x = inputs[i]
    assert isinstance(x, Tensor)
    inputs[i] = ReturnnReinterpretSetAxes(dims_by_key=axes_remap)(x)

  if "B" in torch_special_axes:
    torch_batch_axes = torch_special_axes["B"]
    assert len(torch_batch_axes) == 1
    torch_batch_axis = list(torch_batch_axes)[0]
    # Now (lastly) get rid of potential broadcast batch dims.
    from .shape import Squeeze
    for i, x in enumerate(inputs):
      if not isinstance(x, Tensor):
        continue
      x = naming.tensors[x]
      assert isinstance(x, TensorEntry)
      if x.returnn_data.batch_ndim == 0:  # ignore scalars
        continue
      if not x.returnn_data.have_batch_axis():
        returnn_batch_axis = x.returnn_axis_from_torch_axis[torch_batch_axis]
        assert x.returnn_data.batch_shape[returnn_batch_axis] == 1  # expected to be broadcast dim
        # In RETURNN, that's not needed, and that also must not be done like that,
        # i.e. broadcasting in the batch dim can only be implicit (i.e. by missing batch-dim).
        # Note that now the number of dimensions do not match anymore!
        # But this should not matter for the use cases of this function.
        inputs[i] = Squeeze(dim=torch_batch_axis)(x.tensor())

  return tuple(inputs)


__all__ = [
  key for (key, value) in sorted(globals().items())
  if not key.startswith("_")
  and getattr(value, "__module__", "") == __name__]
