from typing import Optional, Dict, Tuple, Any, Sequence
from returnn.tf.util.data import Dim, batch_dim, single_step_dim
from ..torch._C import Size, SizeValue


class ReturnnDimTagsProxy:
  """
  When serialized via __repr__, this represents a dict unique_name -> dim tag.
  All usages in the network and extern_data will also get proxies when serialized point to this dict.

  Copied from returnn_common
  """

  def __init__(self):
    self.dim_refs_by_name = {}  # type: Dict[str, ReturnnDimTagsProxy.DimRefProxy]
    self.dim_refs_by_tag = {}  # type: Dict[Dim, ReturnnDimTagsProxy.DimRefProxy]

  def __repr__(self):
    return "\n".join([
      "{",
      *(f"  {key!r}: {value.dim_repr()}," for key, value in self.dim_refs_by_name.items()),
      "}"])

  def copy(self):  # -> ReturnnDimTagsProxy:
    """
    :return: creates a shallow copy
    """
    new = ReturnnDimTagsProxy()
    new.dim_refs_by_name = self.dim_refs_by_name.copy()
    new.dim_refs_by_tag = self.dim_refs_by_tag.copy()
    return new

  def py_code_str(self):
    """
    :return: Python code
    """
    return "".join([
      *(f"{value.py_id_name()} = {value.dim_repr()}\n" for key, value in self.dim_refs_by_name.items()),
      ])

  def dim_ref_repr(self, dim: Dim) -> str:
    """
    :return: for the given dim, Python code which refers to it, via ``dim_tags``
    """
    if dim == batch_dim:
      return "batch_dim"
    if dim == single_step_dim:
      return "single_step_dim"
    if dim.match_priority:
      return f"{self.dim_ref_repr(dim.copy(match_priority=0))}.copy(match_priority={dim.match_priority})"
    if dim.derived_from_op:
      if dim.derived_from_op.kind == "constant":
        return str(dim.derived_from_op.attribs["value"])
      if dim.derived_from_op.kind == "truediv_left":
        assert len(dim.derived_from_op.inputs) == 2
        a, b = dim.derived_from_op.inputs
        return f"{self.dim_ref_repr(a)}.div_left({self.dim_ref_repr(b)})"
      op_str = {"add": "+", "mul": "*", "truediv_right": "//", "floordiv_right": "//"}[dim.derived_from_op.kind]
      return f" {op_str} ".join(self.dim_ref_repr(in_) for in_ in dim.derived_from_op.inputs)
    ref = self.dim_refs_by_tag[dim]
    return ref.py_id_name()

  class DimRefProxy:
    """
    This will be a reference to the global dim_tags __repr__.
    """
    def __init__(self, *, dim: Dim, name: Optional[str], path: Tuple[Any, ...], parent):  # ReturnnDimTagsProxy):
      self.dim = dim
      self.name = name  # None, or valid Python identifier
      self.path = path
      self.parent = parent
      self.debug_idx = len(parent.dim_refs_by_name)

    def __repr__(self):
      return self.ref_repr()

    def ref_repr(self) -> str:
      """ref repr"""
      return self.parent.dim_ref_repr(self.dim)

    def py_id_name(self) -> str:
      """
      :return: valid Python identifier
      """
      assert self.name
      return self.name + "_dim"

    def dim_repr(self):
      """
      Dim repr, used for serialization of all registered dim tags.
      Any derived dims or special dims will not be registered and instead be represented
      with the same derivation referencing other registered dim tags.
      See :func:`ReturnnDimTagsProxy.dim_ref_repr`.
      """
      dim = self.dim
      assert not dim.is_batch_dim()
      assert dim.can_be_used_as_dim()
      assert not dim.derived_from_op
      assert not dim.match_priority
      dimension = dim.dimension
      if isinstance(dimension, SizeValue):
        dimension = int(dimension)
      # We assume FeatureDim, SpatialDim and Dim are imported.
      if dim.kind == Dim.Types.Feature:
        return f"FeatureDim({dim.description!r}, {dimension})"
      if dim.kind == Dim.Types.Spatial:
        if dimension is not None:
          return f"SpatialDim({dim.description!r}, {dimension})"
        else:
          return f"SpatialDim({dim.description!r})"
      # generic fallback
      return f"Dim(kind={dim.kind}, description={dim.description!r}, dimension={dimension})"

  class SetProxy:
    """
    This represents a set but with a predefined order.
    We want a deterministic order in the repr such that the generated code stays deterministic.
    """
    def __init__(self, values: Sequence[Any]):
      self.values = values

    def __repr__(self):
      return f"{{{', '.join(map(repr, self.values))}}}"

  def collect_dim_tags_and_transform_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Go through the config and collect all dim tags, replace them by proxies.

    :return: new config
    """
    import re

    def _sort_key(value):
      if isinstance(value, ReturnnDimTagsProxy.DimRefProxy):
        if value.dim.kind == Dim.Types.Batch:
          return -1
        return value.debug_idx
      return value

    def _unique_name(dim: Dim) -> str:
      assert dim not in self.dim_refs_by_tag
      name_ = dim.description
      name_ = re.sub(r"[^a-zA-Z0-9_]", "_", name_)
      if name_.endswith("_dim"):
        name_ = name_[:-len("_dim")]
      if name_ not in self.dim_refs_by_name:
        return name_
      i = 0
      while True:
        name__ = f"{name_}_{i}"
        if name__ not in self.dim_refs_by_name:
          return name__
        i += 1

    # Cannot use nest because nest does not support sets. Also nest requires them to be sorted.
    def _map(path, value):
      if isinstance(value, SizeValue):
        return int(value)
      if isinstance(value, Dim):
        if value in {batch_dim, single_step_dim}:
          # No need to register this.
          return ReturnnDimTagsProxy.DimRefProxy(dim=value, name=None, path=path, parent=self)
        if value.derived_from_op:
          # Make sure all the inputs are registered.
          for i, child in enumerate(value.derived_from_op.inputs):
            _map(path + (value.derived_from_op.kind, i), child)
          # No need to register this.
          return ReturnnDimTagsProxy.DimRefProxy(dim=value, name=None, path=path, parent=self)
        if value.match_priority != 0:
          _map(path, value.copy(match_priority=0))  # Register the dim tag without match_priority.
          # Now return the custom proxy for the dim tag with match_priority. No need to register this.
          return ReturnnDimTagsProxy.DimRefProxy(dim=value, name=None, path=path, parent=self)
        if value in self.dim_refs_by_tag:
          return self.dim_refs_by_tag[value]
        name = _unique_name(value)
        assert name not in self.dim_refs_by_name
        ref = ReturnnDimTagsProxy.DimRefProxy(dim=value, name=name, path=path, parent=self)
        self.dim_refs_by_name[name] = ref
        self.dim_refs_by_tag[value] = ref
        return ref
      if isinstance(value, dict):
        return {
          _map(path + (key, "key"), key): _map(path + (key, "value"), value_)
          for key, value_ in value.items()}
      if isinstance(value, list):
        return [_map(path + (i,), value_) for i, value_ in enumerate(value)]
      if isinstance(value, tuple) and type(value) is tuple:
        return tuple(_map(path + (i,), value_) for i, value_ in enumerate(value))
      if isinstance(value, tuple) and type(value) is Size:
        return Size(_map(path + (i,), value_) for i, value_ in enumerate(value))
      if isinstance(value, tuple) and type(value) is not tuple:
        # noinspection PyProtectedMember,PyUnresolvedReferences,PyArgumentList
        return type(value)(*(_map(path + (key,), getattr(value, key)) for key in value._fields))
      if isinstance(value, set):
        values = [_map(path + (value,), value_) for value_ in value]
        values.sort(key=_sort_key)  # this should be possible now because it would be some sortable proxies
        return ReturnnDimTagsProxy.SetProxy(values)
      return value

    config = _map((), config)
    return config
