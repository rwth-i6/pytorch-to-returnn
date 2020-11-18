
from __future__ import annotations

import tensorflow as tf
import typing
from typing import Optional, Any, List, TypeVar, Dict, Callable, Iterable, Union
import weakref
from weakref import WeakKeyDictionary, ref
from collections import OrderedDict
from contextlib import contextmanager
import itertools
from returnn.config import Config
from returnn.tf.network import ExternData, TFNetwork
from returnn.tf.layers.basic import InternalLayer, LayerBase, CopyLayer, SubnetworkLayer
from returnn.tf.util.data import Data

if typing.TYPE_CHECKING:
  # Just for typing. Although we also cover traced/wrapped Torch.
  from .torch import Tensor
  from .torch.nn import Module


class TensorEntry:
  tensor: ref[Tensor]
  returnn_data: Optional[Data] = None
  is_param: bool = False
  is_const: bool = False  # e.g. via from_numpy, empty, zeros, etc
  is_input: bool = False  # in TF1 terminology, would be a placeholder
  output_from_modules: List["ModuleEntry"]
  output_from_calls: List["CallEntry"]
  parent_owning_modules: List["ModuleEntry"]  # e.g. param or buffer
  creation_stack_call: Optional[CallEntry]
  names: List["RegisteredName"]

  def __init__(self, tensor: ref[Tensor], creation_stack_call: Optional[CallEntry]):
    self.tensor = tensor
    self.creation_stack_call = creation_stack_call
    self.output_from_modules = []
    self.output_from_calls = []
    self.parent_owning_modules = []
    self.names = []


class CallEntry:
  """
  Can be a module() call, or regular func.
  Note that a module can be called multiple times.
  """
  func: Optional[Callable]
  module: Optional["ModuleEntry"]
  inputs: Optional[List["TensorEntry"]] = None
  outputs: Optional[List["TensorEntry"]] = None
  parent_call: Optional["CallEntry"] = None  # parent in the call stack
  child_calls: List["CallEntry"]
  level: Optional[int] = None
  namespace: Optional["RegisteredName"] = None

  def __init__(self, func: Optional[Callable], module: Optional["ModuleEntry"]):
    self.func = func
    self.module = module
    self.child_calls = []

  def __repr__(self):
    return f"<{self.__class__.__name__} #{self.level} {self.func!r}>"

  def is_module_call(self) -> bool:
    if not self.module:
      return False
    return self.module.module is self.func

  def get_root_call(self) -> "CallEntry":
    entry = self
    while entry.parent_call:
      entry = entry.parent_call
    return entry

  def get_canonical_name(self) -> str:
    """
    Considering the canonical context where this is being used.
    Not an absolute name but relative.
    :return:
    """
    if self.module:
      return self.module.get_canonical_name()
    if self.parent_call:
      prefix = self.parent_call.get_canonical_name() + "_"
    else:
      prefix = ""
    return prefix + self.func.__name__

  def set_outputs(self, outputs: List[Tensor]):
    assert self.outputs is None
    naming = Naming.get_instance()
    entry_outputs = [naming.register_tensor(x) for x in outputs]
    self.outputs = entry_outputs
    for x in entry_outputs:
      x.output_from_calls.append(self)
      if self.module:
        x.output_from_modules.append(self.module)
    if entry_outputs:
      entry_outputs[0].names.append(self.namespace)

  def __enter__(self):
    # Assume via push_func_call
    assert Naming.get_instance().module_call_stack[-1] is self
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    if not exc_type:
      Naming.get_instance().pop_module_call(self)


class _ObjAttr:
  T = TypeVar("T")
  obj: T
  attr: str

  def __init__(self, obj: T, attr: str):
    self.obj = obj
    self.attr = attr


class ModuleEntry:
  module: Module
  level: Optional[int] = None
  calls: List[CallEntry]
  names: List[RegisteredName]
  parent_created_modules: List["ModuleEntry"]
  child_created_modules: List["ModuleEntry"]
  parent_owning_modules: List["ModuleEntry"]
  parent_context_modules: List["ModuleEntry"]

  def __init__(self, module: Module):
    self.module = module
    self.calls = []
    self.names = []
    self.parent_created_modules = []
    self.child_created_modules = []
    self.parent_owning_modules = []
    self.parent_context_modules = []

  def __repr__(self):
    return f"<ModuleEntry {self.module!r}>"

  def get_parent_calling_modules(self) -> List["ModuleEntry"]:
    res = []
    for call in self.calls:
      assert call.module is self
      while call.parent_call:
        call = call.parent_call
        if call.module:
          res.append(call.module)
          break
    return res

  def get_root_owning_module(self) -> "ModuleEntry":
    mod = self
    while mod.parent_owning_modules:
      mod = mod.parent_owning_modules[0]
    return mod

  def get_canonical_name(self) -> str:
    for mod in self.parent_owning_modules:
      if not mod.module.forward:
        prefix = mod.get_canonical_name() + "_"
      else:
        prefix = ""
      for name, child_mod in mod.module.named_children():
        if child_mod is self.module:
          if not prefix and name[:1].isnumeric():
            return f"layer{name}"
          return prefix + name
    prefix = ""
    for mod in reversed(self.parent_context_modules):
      prefix = mod.get_canonical_name() + "_"
      break
    return prefix + self.module.__class__.__name__

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    if not exc_type:
      Naming.get_instance().pop_module_context(self.module)


_SearchType = TypeVar("_SearchType")
_SearchChildsFuncT = Callable[[_SearchType], Iterable[_SearchType]]
_SearchFilterFuncT = Callable[[_SearchType], bool]


def _breadth_first_search(
      root: _SearchType = None, *,
      seed: List[_SearchType] = None,
      childs: _SearchChildsFuncT, filter_elem: Optional[_SearchFilterFuncT] = None):
  visited = set()
  if seed:
    assert root is None
    queue = seed
  else:
    queue = [root]
  while queue:
    next_queue = []
    for elem in queue:
      if elem in visited:
        continue
      visited.add(elem)
      if filter_elem and not filter_elem(elem):
        continue
      yield elem
      next_queue.extend(childs(elem))
    queue = next_queue


def _breadth_first_search_first(
      root: _SearchType, childs: _SearchChildsFuncT, filter_elem: Optional[_SearchFilterFuncT] = None):
  for elem in _breadth_first_search(root=root, childs=childs, filter_elem=filter_elem):
    return elem
  return None


class RegisteredName:
  childs_by_name: OrderedDict[str, "RegisteredName"]
  parent: Optional["RegisteredName"]
  name: Optional[str]  # if parent
  level: int = 0
  modules: List[ModuleEntry]  # can be multiple merged together
  calls: List[CallEntry]  # can be multiple merged together. can be empty if this is some input
  tensor: Optional[TensorEntry] = None  # output from the call
  returnn_ctx: Optional[ReturnnContext] = None

  def __init__(self, *,
               parent: Optional["RegisteredName"] = None, name: Optional[str] = None,
               call: Optional[CallEntry] = None,
               tensor: Optional[TensorEntry] = None):
    self.childs_by_name = OrderedDict()
    self.parent = parent
    if parent:
      assert name
    else:
      assert not name
    self.name = name
    self.modules = []
    self.calls = []
    if call:
      self.assign_call(call)
    if parent:
      self.level = parent.level + 1
    if tensor:
      self.assign_tensor(tensor)
    if not parent:
      self.returnn_ctx = ReturnnContext(parent=None, name=self.name)

  def __repr__(self):
    return f"<{self.__class__.__name__} {self.get_absolute_name()!r}>"

  def get_absolute_name(self):
    names = []
    name_ = self
    while name_.parent:
      names.insert(0, name_.name)
      name_ = name_.parent
    return "/".join(names) if names else ""

  def assign_tensor(self, tensor: TensorEntry):
    assert not self.tensor
    self.tensor = tensor
    if tensor:
      tensor.names.append(self)

  def assign_call(self, call: CallEntry):
    if call.module:
      self.assign_module(call.module)
    assert all(not c.module or c.module.module.forward for c in self.calls)
    if self.calls:
      assert not call.module or call.module.module.forward
    assert not call.namespace
    call.namespace = self
    self.calls.append(call)

  def assign_module(self, module: ModuleEntry):
    if module in self.modules:
      return
    self.modules.append(module)
    module.names.append(self)
    if module.module.forward and not self.returnn_ctx:
      # Need our own returnn ctx / subnet.
      assert self.parent
      self.returnn_ctx = ReturnnContext(parent=self.parent.returnn_ctx, name=self.name)
    if self.returnn_ctx:
      assert all(m.module.forward for m in self.modules)

  def _get_unique_name(self, suggested_name: str) -> str:
    if suggested_name not in self.childs_by_name:
      return suggested_name
    for i in itertools.count(1):
      suggested_name_ = f"{suggested_name}_{i}"
      if suggested_name_ not in self.childs_by_name:
        return suggested_name_

  def register(self, *, suggested_name: str, call: Optional[CallEntry] = None) -> RegisteredName:
    name = self._get_unique_name(suggested_name)
    name_ = RegisteredName(parent=self, name=name, call=call)
    self.childs_by_name[name] = name_
    return name_

  def register_input(self, name: str, tensor: TensorEntry) -> RegisteredName:
    if name in self.childs_by_name:
      assert self.childs_by_name[name].tensor is tensor
      return self.childs_by_name[name]
    assert name not in self.childs_by_name
    name_ = RegisteredName(parent=self, name=name, tensor=tensor)
    # TODO hardcoded defaults
    data_key = "data"
    if not tensor.returnn_data:
      assert self.returnn_ctx
      assert data_key not in self.returnn_ctx.extern_data.data
      assert tensor.tensor().dim() == 3  # assume dense (B,T,D), TODO
      tensor.returnn_data = Data(
        name=data_key, auto_create_placeholders=True, dim=tensor.tensor().shape[-1], available_for_inference=True)
    self.returnn_ctx.extern_data.data[data_key] = tensor.returnn_data
    self.childs_by_name[name] = name_
    return name_

  def name_for_tensor(self, tensor: TensorEntry) -> str:
    for name_ in tensor.names:
      if name_.parent is self:
        assert self.childs_by_name[name_.name] is name_
        return name_.name
    raise KeyError(f"namespace {self!r}: tensor {tensor!r} not found")

  def find_name_for_module(self, module: ModuleEntry) -> Optional[str]:
    for name, child in self.childs_by_name.items():
      if module in child.modules:
        return name
    return None

  def dump(self, prefix=""):
    for name, child in self.childs_by_name.items():
      print(f"{prefix}{name}: {child.calls}")
      child.dump(prefix=f"{prefix}  ")


class ReturnnContext:
  def __init__(self, *, parent: Optional[ReturnnContext] = None, name: Optional[str] = None):
    self.parent = parent
    if parent:
      assert name
      self.config = parent.config
      self.tf_name_scope = parent.network.get_absolute_name_scope_prefix() + LayerBase.cls_get_tf_scope_name(name)
      assert parent.network.extern_data.data
      self._sub_layer = (
        parent.network.add_layer(
          name=name, layer_class=SubnetworkLayer,
          # This is just a placeholder, will be replaced in define_output.
          sources=[parent.network.get_layer("data")],
          subnetwork={"output": {"class": "copy"}}))  # type: SubnetworkLayer
      self._dummy_sub_output = self._sub_layer.subnetwork.layers["output"]
    else:
      self.config = Config({
        # "debug_print_layer_output_template": True,
      })
      self.tf_name_scope = ""
      self._sub_layer = None
      self._dummy_sub_output = None
    self.extern_data = ExternData()
    if self._sub_layer:
      self.network = self._sub_layer.subnetwork
    else:
      assert not parent
      self.network = TFNetwork(extern_data=self.extern_data, config=self.config, name="root")

  def add_layer(self):
    pass

  def define_output(self, layer_name: str) -> LayerBase:
    assert layer_name in self.network.layers
    if "output" in self.network.layers:
      assert self.network.layers["output"] is self._dummy_sub_output
      self.network.layers.pop("output")
      self._dummy_sub_output = None
    self.network.construct_layer({"output": {"class": "copy", "from": layer_name}}, name="output")
    layer = self.network.layers["output"]
    if self._sub_layer:
      self._sub_layer.output = layer.output
    return layer


class Naming:
  tensors: WeakKeyDictionary[Tensor, TensorEntry]
  modules: OrderedDict[Module, ModuleEntry]
  inputs: List[Tensor]
  outputs: List[Tensor]
  module_creation_call_stack: List[ModuleEntry]
  module_context_stack: List[ModuleEntry]
  module_call_stack: List[CallEntry]
  root_func_calls: List[CallEntry]
  _instance: Optional[Naming] = None

  @classmethod
  @contextmanager
  def make_instance(cls) -> Naming:
    assert not cls._instance
    ctx = Naming()
    cls._instance = ctx
    yield ctx
    assert cls._instance is ctx
    cls._instance = None

  @classmethod
  def get_instance(cls) -> Naming:
    assert cls._instance
    return cls._instance

  def __init__(self):
    self.tensors = WeakKeyDictionary()
    self.modules = OrderedDict()
    self.inputs = []
    self.outputs = []
    self.module_creation_call_stack = []
    self.module_context_stack = []
    self.module_call_stack = []
    self.root_func_calls = []
    self.root_namespace = RegisteredName(parent=None)

  def push_module_context(self, module: Module) -> ModuleEntry:
    entry = self.modules[module]
    if self.module_context_stack:
      if self.module_context_stack[-1] not in entry.parent_context_modules:
        entry.parent_context_modules.append(self.module_context_stack[-1])
    self.module_context_stack.append(entry)
    return entry

  def pop_module_context(self, module: Module):
    entry = self.modules[module]
    assert self.module_context_stack[-1] is entry
    self.module_context_stack.pop(-1)

  def push_module_creation(self, module: Module):
    if module not in self.modules:
      entry = ModuleEntry(module=module)
      self.modules[module] = entry
    else:
      entry = self.modules[module]
    if self.module_creation_call_stack:
      recent_entry = self.module_creation_call_stack[-1]
      if recent_entry not in entry.parent_created_modules:
        entry.parent_created_modules.append(recent_entry)
        recent_entry.child_created_modules.append(entry)
    self.module_creation_call_stack.append(entry)
    self.push_module_context(module)

  def pop_module_creation(self, module: Module):
    self.pop_module_context(module)
    assert self.module_creation_call_stack[-1].module is module
    self.module_creation_call_stack.pop(-1)

  def push_module_call(self, *,
                       module: Module, func: Optional[Callable], inputs: List[Tensor]) -> CallEntry:
    module_entry = self.modules[module] if module is not None else None
    entry = CallEntry(func=func, module=module_entry)
    entry.inputs = [self.tensors[x] for x in inputs]
    entry.level = len(self.module_call_stack)
    if self.module_call_stack:
      recent_entry = self.module_call_stack[-1]
      recent_entry.child_calls.append(entry)
      entry.parent_call = recent_entry
      if recent_entry.is_module_call() and recent_entry.module.module.create_returnn_layer_dict:
        raise Exception(f"Not expected to have sub call stacks on module {recent_entry.module.module}")
    else:
      self.root_func_calls.append(entry)
    if entry.parent_call:
      assert entry.parent_call.namespace
      parent_namespace = entry.parent_call.namespace
    else:
      parent_namespace = self.root_namespace
    # Find right parent namespace.
    parents_hierarchy = []
    parent_module = module_entry
    while parent_module not in parent_namespace.modules:
      parents_hierarchy.append(parent_module)
      if not parent_module.parent_owning_modules:
        if parent_module not in self.module_context_stack and self.module_context_stack:
          parent_module = self.module_context_stack[-1]
          continue  # try further
        elif self.module_context_stack.index(parent_module) > 0:
          parent_module = self.module_context_stack[self.module_context_stack.index(parent_module) - 1]
          continue  # try further
        # no parent anymore, so use root namespace in any case
        parent_namespace = self.root_namespace
        break
      parent_module = parent_module.parent_owning_modules[0]  # could do search over all, but just use first for now
    for parent_module in reversed(parents_hierarchy):
      assert parent_module not in parent_namespace.modules
      if parent_module is not module_entry and not parent_module.module.forward:
        # Skip.
        parent_module = module_entry
      if (parent_namespace is self.root_namespace
              and parent_module.module.forward and not parent_module.parent_owning_modules):
        # Special case, somewhat nicer to flatten the namespace for this case.
        self.root_namespace.assign_module(parent_module)
      else:
        name = parent_namespace.find_name_for_module(parent_module)
        if name:
          parent_namespace = parent_namespace.childs_by_name[name]
        else:
          parent_namespace = parent_namespace.register(suggested_name=parent_module.get_canonical_name())
          parent_namespace.assign_module(parent_module)
      assert parent_module in parent_namespace.modules
      if parent_module is module_entry:
        break
    assert parent_module in parent_namespace.modules and parent_module is module_entry
    namespace = parent_namespace
    assert module_entry in namespace.modules
    if func is module and not module.forward:
      assert namespace.parent
      for x in entry.inputs:
        assert isinstance(x, TensorEntry)
        if not x.names and x.is_param:
          from .torch.nn.modules import Variable
          mod = Variable(param=x.tensor())
          res = mod()
          # TODO
        name = namespace.parent.name_for_tensor(x)
        assert name
    namespace.assign_call(entry)
    self.module_call_stack.append(entry)
    assert entry.namespace
    if module:
      self.push_module_context(module)
    return entry

  def pop_module_call(self, call: CallEntry):
    if call.module:
      self.pop_module_context(call.module.module)
    assert self.module_call_stack[-1] is call
    self.module_call_stack.pop(-1)

  def register_module_child_attr(self, parent: Module, attr: str, child: Union[Module, Tensor]):
    assert getattr(parent, attr) is child
    parent_entry = self.modules[parent]
    if child in self.modules:
      child_entry = self.modules[child]
    else:
      assert child in self.tensors
      child_entry = self.tensors[child]
      assert isinstance(child_entry, TensorEntry)
      for parent_param in parent.parameters(recurse=False):
        if parent_param is child:
          child_entry.is_param = True
          break
    if parent_entry not in child_entry.parent_owning_modules:
      child_entry.parent_owning_modules.append(parent_entry)

  def register_tensor(self, tensor: Tensor) -> TensorEntry:
    if tensor not in self.tensors:
      self.tensors[tensor] = TensorEntry(
        tensor=ref(tensor), creation_stack_call=self.module_call_stack[-1] if self.module_call_stack else None)
    return self.tensors[tensor]

  def register_input(self, tensor: Tensor):
    entry = self.register_tensor(tensor)
    entry.is_input = True
    self.inputs.append(tensor)
    # "data" is a special layer name in RETURNN, representing input data
    self.root_namespace.register_input(name="data", tensor=entry)

  @staticmethod
  def _register_call_names(root: RegisteredName, calls: List[CallEntry]):
    for call in calls:
      child = root.register(suggested_name=call.get_canonical_name(), call=call)
      Naming._register_call_names(child, call.child_calls)

  def register_output(self, tensor: Tensor):
    assert tensor in self.tensors
    entry = self.tensors[tensor]
    assert isinstance(entry, TensorEntry)
    assert not entry.is_param and not entry.is_const and not entry.is_input  # not implemented, although simple...
    self.outputs.append(tensor)

    self.root_namespace.dump()
