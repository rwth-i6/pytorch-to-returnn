
from typing import Optional, Tuple, Any, List, Dict
from returnn.tf.util.data import Data, DimensionTag
from .module import Module
from ...tensor import Tensor
from ....naming import Naming, TensorEntry


class BinaryOperator(Module):
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


class Reciprocal(Module):
  """
  1/x or 1/max(eps,x)
  """
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
  pass  # TODO


class ReturnnReinterpretSameSizeAs(Module):
  """
  Used by :func:`_unify_tensor_dyn_axes`.
  """
  def create_returnn_layer_dict(self, input: Tensor, same_as: Tensor) -> Dict[str, Any]:
    return {
      "class": "reinterpret_data",
      "from": self._get_input_layer_name(input),
      "size_base": self._get_input_layer_name(same_as)}


def _unify_tensor_dyn_axes(*inputs: Tensor) -> Tuple[Tensor]:
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
