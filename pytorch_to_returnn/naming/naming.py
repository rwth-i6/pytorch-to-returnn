
from __future__ import annotations
from tensorflow.python.util import nest
from typing import Generator, List, Optional, Union, Dict, Any, Tuple
from contextlib import contextmanager
from collections import OrderedDict
from weakref import ref, WeakKeyDictionary
import numpy
from returnn.tf.util.data import Data
from returnn.tf.layers.basic import LayerBase
from . import _types
from . import tensor as _tensor
from . import module as _module
from . import call as _call
from . import namespace as _namespace


class Naming:
  tensors: WeakKeyDictionary[_types.Tensor, _tensor.TensorEntry]
  const_tensor_cache: List[_types.Tensor]
  modules: OrderedDict[_types.Module, _module.ModuleEntry]
  inputs: List[_types.Tensor]
  outputs: List[_types.Tensor]
  module_creation_stack: List[_module.ModuleEntry]
  module_apply_stack: List[_module.ModuleEntry]
  module_context_stack: List[_module.ModuleEntry]
  module_call_stack: List[_call.CallEntry]
  root_func_calls: List[_call.CallEntry]
  _default_instance: Optional[Naming] = None
  _instance_stack: List[Naming] = []

  @classmethod
  @contextmanager
  def make_instance(cls, **kwargs) -> Generator[Naming]:
    ctx = Naming(**kwargs)
    cls._instance_stack.append(ctx)
    yield ctx
    assert cls._instance_stack[-1] is ctx
    cls._instance_stack.pop(-1)

  @classmethod
  def get_instance(cls) -> Naming:
    if cls._instance_stack:
      return cls._instance_stack[-1]
    if not cls._default_instance:
      cls._default_instance = Naming()
    return cls._default_instance

  def __init__(self, *,
               wrap_to_returnn_enabled: bool = True,
               keep_orig_module_io_tensors: bool = True,
               import_params_from_torch_namespace: Optional[Naming] = None,
               validate_allclose_kwargs: Optional[Dict[str, Any]] = None,
               ):
    """
    :param wrap_to_returnn_enabled: Will construct corresponding RETURNN layers.
    :param keep_orig_module_io_tensors: Keeps references to the original (or wrapped) torch.Tensor instances
       for all module calls. This will need extra memory.
    :param import_params_from_torch_namespace: If given, we try to import params.
    :param validate_allclose_kwargs: for numpy.allclose
    """
    self.wrap_to_returnn_enabled = wrap_to_returnn_enabled
    self.keep_orig_module_io_tensors = keep_orig_module_io_tensors
    self.import_params_from_torch_namespace = import_params_from_torch_namespace
    if validate_allclose_kwargs is None:
      # PyTorch uses some different algos, e.g. different convolution,
      # which leads to quite huge relative differences (for values close to 0.0).
      validate_allclose_kwargs = dict(rtol=0, atol=5e-4)
    self.validate_allclose_kwargs = validate_allclose_kwargs
    self.tensors = WeakKeyDictionary()
    self.const_tensor_cache = []
    self.modules = OrderedDict()
    self.inputs = []
    self.outputs = []
    self.module_context_stack = []
    self.module_creation_stack = []
    self.module_apply_stack = []
    self.module_call_stack = []
    self.root_func_calls = []
    self.root_namespace = _namespace.RegisteredName(
      parent=None, wrap_to_returnn_enabled=wrap_to_returnn_enabled, is_subnet=True)
    self.tmp_eager_root_namespace = self.root_namespace.register_sub_net(suggested_name=".tmp_root")

  @contextmanager
  def push_module_creation(self, module: _types.Module) -> _module.ModuleEntry:
    assert module not in self.modules
    entry = _module.ModuleEntry(module=module)
    self.modules[module] = entry
    self.module_creation_stack.append(entry)
    with self.push_module_context(module):
      yield entry
    assert self.module_creation_stack[-1] is entry
    self.module_creation_stack.pop(-1)

  @contextmanager
  def push_module_apply(self, module: _types.Module) -> _module.ModuleEntry:
    entry = self.modules[module]
    self.module_apply_stack.append(entry)
    with self.push_module_context(module):
      yield entry
    assert self.module_apply_stack[-1] is entry
    self.module_apply_stack.pop(-1)

  def push_module_context(self, module: _types.Module) -> _module.ModuleEntry:
    entry = self.modules[module]
    if self.module_context_stack:
      prev_top = self.module_context_stack[-1]
      if prev_top not in entry.parent_context_modules and prev_top.module is not module:
        entry.parent_context_modules.append(prev_top)
    self.module_context_stack.append(entry)
    return entry

  def pop_module_context(self, module: _types.Module):
    entry = self.modules[module]
    assert self.module_context_stack[-1] is entry
    self.module_context_stack.pop(-1)

  def _prepare_module_call_returnn_inputs(self, call: _call.CallEntry):
    """
    It means this module has no forward, i.e. it is wrapped as RETURNN layer.
    We might need to make some inputs available, which are not available yet,
    e.g. constants, params, etc.
    """
    if not self.wrap_to_returnn_enabled:
      return
    call_parent_namespace = call.namespace.parent or self.root_namespace
    for x in call.inputs_flat:
      if x is None:
        continue
      assert isinstance(x, _tensor.TensorEntry)
      names = [name_ for name_ in x.names if name_.parent is call_parent_namespace]
      if names:
        continue
      if x.is_param:
        assert x.returnn_data
        assert x.returnn_data.placeholder is None
        from pytorch_to_returnn.torch.nn.modules import Variable
        param_name = x.get_canonical_name()
        if x.returnn_data.name == "_unnamed_param":
          x.returnn_data.name = f"param:{param_name}"
        parent_mod = x.get_canonical_parent_module()
        prefix = (parent_mod.get_canonical_name() + "_") if parent_mod else ""
        mod = Variable(param=x.tensor())
        self.modules[mod].canonical_name = prefix + param_name
        res = mod()
        res_tensor = self.tensors[res]
        assert isinstance(res_tensor, _tensor.TensorEntry)
        assert len(res_tensor.names) == 1
        assert res_tensor.returnn_data.placeholder is not None
        x.returnn_data.placeholder = res_tensor.returnn_data.placeholder
      elif not x.output_from_calls or x.is_const:
        # Assume this is a constant.
        const_name = x.get_canonical_name(fallback="unnamed_const")
        tensor = x.tensor()
        if not x.returnn_data:
          x.returnn_data = Data(
            name=f"const:{const_name}", shape=tensor.shape, dtype=tensor.dtype.name,
            batch_dim_axis=None, time_dim_axis=None)
          x.returnn_axis_from_torch_axis = {i: i for i in range(len(tensor.shape))}
        parent_mod = x.get_canonical_parent_module()
        prefix = (parent_mod.get_canonical_name() + "_") if parent_mod else ""
        from pytorch_to_returnn.torch.nn.modules import Constant
        mod = Constant(value=tensor)
        self.modules[mod].canonical_name = prefix + const_name
        res = mod()
        res_tensor = self.tensors[res]
        assert isinstance(res_tensor, _tensor.TensorEntry)
        assert res_tensor.returnn_data.placeholder is not None
        x.returnn_data.placeholder = res_tensor.returnn_data.placeholder
        x.is_const = True
      else:
        raise Exception(f"Cannot handle tensor {x}, via {x.output_from_calls} ...")

  def push_module_call(self, *, module: _types.Module,
                       inputs_args: Tuple[Union[_types.Tensor, Any], ...],
                       inputs_kwargs: Dict[str, Union[_types.Tensor, Any]]) -> _call.CallEntry:
    module_entry = self.modules[module]
    assert isinstance(module_entry, _module.ModuleEntry)
    entry = _call.CallEntry(module=module_entry)
    inputs_flat = nest.flatten((inputs_args, inputs_kwargs))
    if self.keep_orig_module_io_tensors:
      entry.orig_inputs_args = inputs_args
      entry.orig_inputs_kwargs = inputs_kwargs
      entry.orig_inputs_flat = inputs_flat
    entry.inputs_flat = [self._make_tensor(x) for x in inputs_flat]
    entry.inputs_args, entry.inputs_kwargs = nest.pack_sequence_as(
      structure=(inputs_args, inputs_kwargs), flat_sequence=entry.inputs_flat)
    entry.level = len(self.module_call_stack)
    if self.module_call_stack:
      recent_entry = self.module_call_stack[-1]
      recent_entry.child_calls.append(entry)
      entry.parent_call = recent_entry
    else:
      self.root_func_calls.append(entry)
    if self.module_creation_stack or self.module_apply_stack:
      # This is likely for parameter initialization, or setting up constants.
      # Keep this in a separate namespace.
      root_namespace = self.tmp_eager_root_namespace
    else:
      root_namespace = self.root_namespace
    if entry.parent_call:
      assert entry.parent_call.namespace
      parent_namespace = entry.parent_call.namespace
      # Get the parent namespace, which is a subnetwork.
      while not parent_namespace.is_subnetwork():
        assert parent_namespace.parent
        parent_namespace = parent_namespace.parent
    else:
      parent_namespace = root_namespace
    assert parent_namespace.is_subnetwork()
    if parent_namespace is root_namespace and _flatten_namespace_for_mod(module_entry):
      # Special case, somewhat nicer to flatten the namespace for this case.
      root_namespace.assign_call(entry)
      namespace = root_namespace
    else:
      namespace = parent_namespace.register_sub_call(entry)
    assert module_entry in namespace.modules
    self.module_call_stack.append(entry)
    assert entry.namespace
    self._prepare_module_call_returnn_inputs(entry)
    return entry

  def pop_module_call(self, call: _call.CallEntry):
    assert self.module_call_stack[-1] is call
    self.module_call_stack.pop(-1)

  def register_module_child_attr(self, parent: _types.Module, attr: str, child: Union[_types.Module, _types.Tensor]):
    assert getattr(parent, attr) is child
    parent_entry = self.modules[parent]
    if child in self.modules:
      child_entry = self.modules[child]
      assert isinstance(child_entry, _module.ModuleEntry)
    else:
      assert child in self.tensors
      child_entry = self.tensors[child]
      assert isinstance(child_entry, _tensor.TensorEntry)
      for parent_param in parent.parameters(recurse=False):
        if parent_param is child:
          child_entry.is_param = True
          break
    if (parent_entry, attr) not in child_entry.parent_owning_modules:
      child_entry.parent_owning_modules.append((parent_entry, attr))

  def register_tensor(self, tensor: _types.Tensor) -> _tensor.TensorEntry:
    if tensor not in self.tensors:
      self.tensors[tensor] = _tensor.TensorEntry(
        tensor=ref(tensor),
        creation_stack_call=self.module_call_stack[-1] if self.module_call_stack else None,
        module_context_stack=list(self.module_context_stack))
    return self.tensors[tensor]

  def _make_tensor(self, x: Union[_types.Tensor, int, float, numpy.number, numpy.ndarray]
                   ) -> Optional[_tensor.TensorEntry]:
    if x is None:
      return None
    if not self.wrap_to_returnn_enabled:
      if x in self.tensors:
        return self.tensors[x]
      return None
    from pytorch_to_returnn.torch import from_numpy, Tensor
    if isinstance(x, (int, float, numpy.number, numpy.ndarray)):
      x = from_numpy(x)
      self.const_tensor_cache.append(x)
    assert isinstance(x, Tensor)
    return self.tensors[x]

  def register_input(self, tensor: _types.Tensor, returnn_data: Data) -> Data:
    entry = self.register_tensor(tensor)
    entry.is_input = True
    entry.is_const = False
    self.inputs.append(tensor)
    assert tensor.dim() == returnn_data.batch_ndim
    assert all([dim in {tensor.shape[i], None} for i, dim in enumerate(returnn_data.batch_shape)])
    entry.returnn_data = Data(
      name=returnn_data.name, auto_create_placeholders=True,
      sparse=returnn_data.sparse,
      dim=returnn_data.dim,
      shape=returnn_data.shape,
      batch_dim_axis=returnn_data.batch_dim_axis,
      time_dim_axis=returnn_data.time_dim_axis,
      feature_dim_axis=returnn_data.feature_dim_axis_or_unspecified,
      available_for_inference=True)
    entry.returnn_axis_from_torch_axis = {i: i for i in range(returnn_data.batch_ndim)}
    self.root_namespace.register_input(tensor=entry)
    assert entry.returnn_data
    return entry.returnn_data

  def register_output(self, tensor: _types.Tensor) -> _tensor.TensorEntry:
    assert tensor in self.tensors
    entry = self.tensors[tensor]
    assert isinstance(entry, _tensor.TensorEntry)
    assert not entry.is_param and not entry.is_const and not entry.is_input  # not implemented, although simple...
    self.outputs.append(tensor)
    if self.wrap_to_returnn_enabled:
      self.root_namespace.register_returnn_subnet_output(entry)
    return entry

  def get_module_abs_name(self, module: _types.Module) -> str:
    parts = []
    mod_entry = self.modules[module]
    while True:
      if mod_entry in self.root_namespace.modules:
        break
      name = self.root_namespace.find_name_for_module(mod_entry)
      if name:
        parts.append(name)
        break
      if mod_entry.parent_owning_modules:
        mod_entry, name = mod_entry.parent_owning_modules[0]
        parts.append(name)
        continue
      raise Exception(f"no name for mod {module}")
    abs_name = ".".join(reversed(parts))
    checked_mod = self.get_module_by_abs_name(abs_name)
    assert checked_mod is module
    return abs_name

  def get_module_by_abs_name(self, name: str) -> _types.Module:
    namespace = self.root_namespace
    if name:
      for part_name in name.split("."):
        if part_name[:1].isnumeric():
          part_name = f"layer{part_name}"
        namespace = namespace.childs_by_name[part_name]
    assert namespace.modules
    return namespace.modules[0].module

  def get_module_call_idx(self, *, module: _types.Module, call: _call.CallEntry) -> int:
    mod_entry = self.modules[module]
    assert call in mod_entry.calls
    return mod_entry.calls.index(call)

  def get_module_calls(self, module: _types.Module) -> List[_call.CallEntry]:
    return self.modules[module].calls

  def get_root_module_calls(self) -> OrderedDict[str, _call.CallEntry]:
    d = OrderedDict()
    for name, sub in self.root_namespace.childs_by_name.items():
      if sub.calls:
        d[name] = sub.calls[0]
    return d

  def get_modules_with_params_by_abs_name(self) -> OrderedDict[str, _types.Module]:
    d = OrderedDict()
    _visited = set()

    def visit(namespace: _namespace.RegisteredName, prefix: str):
      for mod in namespace.modules:
        if mod.module in _visited:
          return
        if not list(mod.module.parameters(recurse=False)):
          continue
        assert prefix and prefix.endswith(".") and len(namespace.modules) == 1
        d[prefix[:-1]] = mod.module
        _visited.add(mod.module)
      for name, sub in namespace.childs_by_name.items():
        if name.startswith("."):
          continue  # skip hidden sub spaces
        visit(namespace=sub, prefix=f"{prefix}{name}.")

    visit(namespace=self.root_namespace, prefix="")
    return d

  def get_returnn_layer_from_module(self, module: _types.Module) -> LayerBase:
    assert self.wrap_to_returnn_enabled
    entry = self.modules[module]
    assert entry.calls
    call = entry.calls[0]
    assert call.returnn_layer
    return call.returnn_layer


def _flatten_namespace_for_mod(mod_entry: _module.ModuleEntry) -> bool:
  if mod_entry.parent_owning_modules:
    # Use the parent module.
    return False
  mod = mod_entry.module
  if not mod.has_torch_forward():
    # For RETURNN wrapped modules, e.g. it means that it has no forward, but wraps to a RETURNN layer.
    # So, keep this as a separate item, do not flatten it.
    return False
  return True
