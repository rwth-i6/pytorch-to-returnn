
from __future__ import annotations
from typing import Generator, List, Optional, Union
from contextlib import contextmanager
from collections import OrderedDict
from weakref import ref, WeakKeyDictionary
import numpy
from returnn.tf.util.data import Data
from returnn.tf.layers.basic import LayerBase
from ._types import Tensor, Module
from .tensor import TensorEntry
from .module import ModuleEntry
from .call import CallEntry
from .namescope import RegisteredName


class Naming:
  tensors: WeakKeyDictionary[Tensor, TensorEntry]
  const_tensor_cache: List[Tensor]
  modules: OrderedDict[Module, ModuleEntry]
  inputs: List[Tensor]
  outputs: List[Tensor]
  module_creation_stack: List[ModuleEntry]
  module_apply_stack: List[ModuleEntry]
  module_context_stack: List[ModuleEntry]
  module_call_stack: List[CallEntry]
  root_func_calls: List[CallEntry]
  _instance: Optional[Naming] = None

  @classmethod
  @contextmanager
  def make_instance(cls, **kwargs) -> Generator[Naming]:
    assert not cls._instance
    ctx = Naming(**kwargs)
    cls._instance = ctx
    yield ctx
    assert cls._instance is ctx
    cls._instance = None

  @classmethod
  def get_instance(cls) -> Naming:
    assert cls._instance
    return cls._instance

  def __init__(self, *,
               wrap_to_returnn_enabled: bool,
               keep_orig_module_io_tensors: bool,
               import_params_from_torch_namespace: Optional[Naming] = None
               ):
    """
    :param wrap_to_returnn_enabled: Will construct corresponding RETURNN layers.
    :param keep_orig_module_io_tensors: Keeps references to the original (or wrapped) torch.Tensor instances
       for all module calls. This will need extra memory.
    :param import_params_from_torch_namespace: If given, we try to import params.
    """
    self.wrap_to_returnn_enabled = wrap_to_returnn_enabled
    self.keep_orig_module_io_tensors = keep_orig_module_io_tensors
    self.import_params_from_torch_namespace = import_params_from_torch_namespace
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
    self.root_namespace = RegisteredName(parent=None, wrap_to_returnn_enabled=wrap_to_returnn_enabled)
    self.tmp_eager_root_namespace = self.root_namespace.register(suggested_name=".tmp_root")

  @contextmanager
  def push_module_creation(self, module: Module) -> ModuleEntry:
    assert module not in self.modules
    entry = ModuleEntry(module=module)
    self.modules[module] = entry
    self.module_creation_stack.append(entry)
    with self.push_module_context(module):
      yield entry
    assert self.module_creation_stack[-1] is entry
    self.module_creation_stack.pop(-1)

  @contextmanager
  def push_module_apply(self, module: Module) -> ModuleEntry:
    entry = self.modules[module]
    self.module_apply_stack.append(entry)
    with self.push_module_context(module):
      yield entry
    assert self.module_apply_stack[-1] is entry
    self.module_apply_stack.pop(-1)

  def push_module_context(self, module: Module) -> ModuleEntry:
    entry = self.modules[module]
    if self.module_context_stack:
      prev_top = self.module_context_stack[-1]
      if prev_top not in entry.parent_context_modules and prev_top.module is not module:
        entry.parent_context_modules.append(prev_top)
    self.module_context_stack.append(entry)
    return entry

  def pop_module_context(self, module: Module):
    entry = self.modules[module]
    assert self.module_context_stack[-1] is entry
    self.module_context_stack.pop(-1)

  def _prepare_module_call_returnn_inputs(self, call: CallEntry):
    """
    It means this module has no forward, i.e. it is wrapped as RETURNN layer.
    We might need to make some inputs available, which are not available yet,
    e.g. constants, params, etc.
    """
    if not self.wrap_to_returnn_enabled:
      return
    for x in call.inputs:
      if x is None:
        continue
      assert isinstance(x, TensorEntry)
      if not x.names:
        if x.is_param:
          assert x.returnn_data
          assert x.returnn_data.placeholder is None
          from .torch.nn.modules import Variable
          param_name = x.get_canonical_name()
          if x.returnn_data.name == "_unnamed_param":
            x.returnn_data.name = f"param:{param_name}"
          parent_mod = x.get_canonical_parent_module()
          prefix = (parent_mod.get_canonical_name() + "_") if parent_mod else ""
          mod = Variable(param=x.tensor())
          self.modules[mod].canonical_name = prefix + param_name
          res = mod()
          res_tensor = self.tensors[res]
          assert isinstance(res_tensor, TensorEntry)
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
            x.returnn_axis_to_torch_axis = {i: i for i in range(len(tensor.shape))}
          parent_mod = x.get_canonical_parent_module()
          prefix = (parent_mod.get_canonical_name() + "_") if parent_mod else ""
          from .torch.nn.modules import Constant
          mod = Constant(value=tensor)
          self.modules[mod].canonical_name = prefix + const_name
          res = mod()
          res_tensor = self.tensors[res]
          assert isinstance(res_tensor, TensorEntry)
          assert len(res_tensor.names) == 1
          assert res_tensor.returnn_data.placeholder is not None
          x.returnn_data.placeholder = res_tensor.returnn_data.placeholder
          x.is_const = True
        else:
          raise Exception(f"Cannot handle tensor {x}, via {x.output_from_calls} ...")

  def push_module_call(self, *, module: Module, inputs: List[Tensor]) -> CallEntry:
    module_entry = self.modules[module]
    entry = CallEntry(module=module_entry)
    if self.keep_orig_module_io_tensors:
      entry.orig_inputs = inputs
    entry.inputs = [self._make_tensor(x) for x in inputs]
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
      if self.wrap_to_returnn_enabled:
        while not parent_namespace.returnn_ctx:  # e.g. if call within another module non-forward call
          assert parent_namespace.parent
          parent_namespace = parent_namespace.parent
    else:
      parent_namespace = root_namespace
    # Find right parent namespace.
    parents_hierarchy = []
    parent_module = module_entry
    while parent_module not in parent_namespace.modules and parent_module not in parents_hierarchy:
      parents_hierarchy.append(parent_module)
      if not parent_module.parent_owning_modules:
        if parent_module not in self.module_context_stack and self.module_context_stack:
          parent_module = self.module_context_stack[-1]
          continue  # try further
        elif parent_module in self.module_context_stack and self.module_context_stack.index(parent_module) > 0:
          parent_module = self.module_context_stack[self.module_context_stack.index(parent_module) - 1]
          continue  # try further
        # no parent anymore, so use root namespace in any case
        parent_namespace = root_namespace
        break
      parent_module = parent_module.parent_owning_modules[0][0]  # could do search over all, but just use first for now
    for parent_module in reversed(parents_hierarchy):
      assert parent_module not in parent_namespace.modules
      if parent_module is not module_entry and not parent_module.module.has_torch_forward():
        # Skip.
        parent_module = module_entry
      if parent_namespace is root_namespace and _flatten_namespace_for_mod(parent_module):
        # Special case, somewhat nicer to flatten the namespace for this case.
        root_namespace.assign_module(parent_module)
      else:
        name = parent_namespace.find_name_for_module(parent_module)
        if name:
          parent_namespace = parent_namespace.childs_by_name[name]
        else:
          parent_namespace = parent_namespace.register(
            suggested_name=parent_module.get_canonical_name(parent_namespace=parent_namespace))
          parent_namespace.assign_module(parent_module)
      assert parent_module in parent_namespace.modules
      if parent_module is module_entry:
        break
    assert parent_module in parent_namespace.modules and parent_module is module_entry
    namespace = parent_namespace
    assert module_entry in namespace.modules
    namespace.assign_call(entry)
    self.module_call_stack.append(entry)
    assert entry.namespace
    self._prepare_module_call_returnn_inputs(entry)
    return entry

  def pop_module_call(self, call: CallEntry):
    assert self.module_call_stack[-1] is call
    self.module_call_stack.pop(-1)

  def register_module_child_attr(self, parent: Module, attr: str, child: Union[Module, Tensor]):
    assert getattr(parent, attr) is child
    parent_entry = self.modules[parent]
    if child in self.modules:
      child_entry = self.modules[child]
      assert isinstance(child_entry, ModuleEntry)
    else:
      assert child in self.tensors
      child_entry = self.tensors[child]
      assert isinstance(child_entry, TensorEntry)
      for parent_param in parent.parameters(recurse=False):
        if parent_param is child:
          child_entry.is_param = True
          break
    if (parent_entry, attr) not in child_entry.parent_owning_modules:
      child_entry.parent_owning_modules.append((parent_entry, attr))

  def register_tensor(self, tensor: Tensor) -> TensorEntry:
    if tensor not in self.tensors:
      self.tensors[tensor] = TensorEntry(
        tensor=ref(tensor),
        creation_stack_call=self.module_call_stack[-1] if self.module_call_stack else None,
        module_context_stack=list(self.module_context_stack))
    return self.tensors[tensor]

  def _make_tensor(self, x: Union[Tensor, int, float, numpy.number, numpy.ndarray]) -> Optional[TensorEntry]:
    if x is None:
      return None
    if not self.wrap_to_returnn_enabled:
      if x in self.tensors:
        return self.tensors[x]
      return None
    from .torch import from_numpy, Tensor
    if isinstance(x, (int, float, numpy.number, numpy.ndarray)):
      x = from_numpy(x)
      self.const_tensor_cache.append(x)
    assert isinstance(x, Tensor)
    return self.tensors[x]

  def register_input(self, tensor: Tensor, returnn_data: Data) -> Data:
    entry = self.register_tensor(tensor)
    entry.is_input = True
    self.inputs.append(tensor)
    assert tensor.dim() == returnn_data.batch_ndim
    assert all([dim in {tensor.shape[i], None} for i, dim in enumerate(returnn_data.batch_shape)])
    entry.returnn_data = Data(
      name=returnn_data.name, auto_create_placeholders=True,
      dim=returnn_data.dim,
      shape=returnn_data.shape,
      batch_dim_axis=returnn_data.batch_dim_axis,
      time_dim_axis=returnn_data.time_dim_axis,
      feature_dim_axis=returnn_data.feature_dim_axis_or_unspecified,
      available_for_inference=True)
    entry.returnn_axis_to_torch_axis = {i: i for i in range(returnn_data.batch_ndim)}
    # "data" is a special layer name in RETURNN, representing input data
    self.root_namespace.register_input(tensor=entry)
    assert entry.returnn_data
    return entry.returnn_data

  def register_output(self, tensor: Tensor) -> TensorEntry:
    assert tensor in self.tensors
    entry = self.tensors[tensor]
    assert isinstance(entry, TensorEntry)
    assert not entry.is_param and not entry.is_const and not entry.is_input  # not implemented, although simple...
    self.outputs.append(tensor)
    if self.wrap_to_returnn_enabled:
      self.root_namespace.register_returnn_subnet_output(entry)
    return entry

  def get_module_abs_name(self, module: Module) -> str:
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

  def get_module_by_abs_name(self, name: str) -> Module:
    namespace = self.root_namespace
    if name:
      for part_name in name.split("."):
        if part_name[:1].isnumeric():
          part_name = f"layer{part_name}"
        namespace = namespace.childs_by_name[part_name]
    assert namespace.modules
    return namespace.modules[0].module

  def get_module_call_idx(self, *, module: Module, call: CallEntry) -> int:
    mod_entry = self.modules[module]
    assert call in mod_entry.calls
    return mod_entry.calls.index(call)

  def get_module_calls(self, module: Module) -> List[CallEntry]:
    return self.modules[module].calls

  def get_root_module_calls(self) -> OrderedDict[str, CallEntry]:
    d = OrderedDict()
    for name, sub in self.root_namespace.childs_by_name.items():
      if sub.calls:
        d[name] = sub.calls[0]
    return d

  def get_modules_with_params_by_abs_name(self) -> OrderedDict[str, Module]:
    d = OrderedDict()
    _visited = set()

    def visit(namespace: RegisteredName, prefix: str):
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

  def get_returnn_layer_from_module(self, module: Module) -> LayerBase:
    assert self.wrap_to_returnn_enabled
    entry = self.modules[module]
    assert entry.calls
    call = entry.calls[0]
    assert call.returnn_layer
    return call.returnn_layer


def _flatten_namespace_for_mod(mod_entry: ModuleEntry) -> bool:
  if mod_entry.parent_owning_modules:
    # Use the parent module.
    return False
  mod = mod_entry.module
  if not list(mod.children()):
    # For RETURNN wrapped modules, e.g. it means that it has no forward, but wraps to a RETURNN layer.
    # But more generally, if there are no children modules, we assume this directly does some operation.
    # So, keep this as a separate item, do not flatten it.
    return False
  return True
