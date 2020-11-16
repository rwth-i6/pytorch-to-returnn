
from __future__ import annotations

import typing
from typing import Optional, Any, List, TypeVar, Dict, Callable, Iterable, Union
import weakref
from weakref import WeakKeyDictionary, ref
from collections import OrderedDict
import itertools
from returnn.config import Config
from returnn.tf.network import ExternData, TFNetwork
from returnn.tf.util.data import Data

if typing.TYPE_CHECKING:
  # Just for typing. Although we also cover traced/wrapped Torch.
  from .torch import Tensor
  from .torch.nn import Module


class TensorEntry:
  tensor: ref[Tensor]
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
  func: Callable
  module: Optional["ModuleEntry"]
  inputs: Optional[List["TensorEntry"]] = None
  outputs: Optional[List["TensorEntry"]] = None
  parent_call: Optional["CallEntry"] = None  # parent in the call stack
  child_calls: List["CallEntry"]
  level: Optional[int] = None
  namespace: Optional["RegisteredName"] = None

  def __init__(self, func: Callable, module: Optional["ModuleEntry"]):
    self.func = func
    self.module = module
    self.child_calls = []

  def __repr__(self):
    return f"<{self.__class__.__name__} #{self.level} {self.func!r}>"

  def get_root_call(self) -> "CallEntry":
    entry = self
    while entry.parent_call:
      entry = entry.parent_call
    return entry

  def get_canonical_name(self) -> str:
    if self.module:
      return self.module.get_canonical_name()
    return self.func.__name__


class _ObjAttr:
  T = TypeVar("T")
  obj: T
  attr: str

  def __init__(self, obj: T, attr: str):
    self.obj = obj
    self.attr = attr


class ModuleEntry:
  module: Module
  name: Optional[str] = None
  level: Optional[int] = None
  calls: List[CallEntry]
  parent_created_modules: List["ModuleEntry"]
  child_created_modules: List["ModuleEntry"]
  parent_owning_modules: List["ModuleEntry"]

  def __init__(self, module: Module):
    self.module = module
    self.calls = []
    self.parent_created_modules = []
    self.child_created_modules = []
    self.parent_owning_modules = []

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
    if self.parent_owning_modules:
      for mod in self.parent_owning_modules:
        for name, child_mod in mod.module.named_children():
          if child_mod is self.module:
            if name[:1].isnumeric():
              return f"layer{name}"
            return name
    return self.module.__class__.__name__


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
  items: List[CallEntry]  # can be multiple merged together. can be empty if this is some input
  tensor: Optional[TensorEntry] = None
  returnn_ctx: Optional[ReturnnContext] = None

  def __init__(self, *,
               parent: Optional["RegisteredName"] = None, name: str = None, tensor: Optional[TensorEntry] = None):
    self.childs_by_name = OrderedDict()
    self.parent = parent
    if parent:
      assert name
    else:
      assert not name
    self.name = name
    self.items = []
    if parent:
      self.level = parent.level + 1
    self.tensor = tensor
    if tensor:
      tensor.names.append(self)
      assert parent and parent.returnn_ctx
      # TODO hardcoded defaults
      data_key = "data"
      assert data_key not in parent.returnn_ctx.extern_data.data
      assert tensor.tensor().dim() == 3  # assume dense (B,T,D), TODO
      data = Data(
        name=data_key, auto_create_placeholders=True, dim=tensor.tensor().shape[-1], available_for_inference=True)
      parent.returnn_ctx.extern_data.data[data_key] = data
    else:
      self.returnn_ctx = ReturnnContext(parent=parent.returnn_ctx if parent else None, name=name)

  def __repr__(self):
    return f"<{self.__class__.__name__} {self.get_absolute_name()!r}>"

  def get_absolute_name(self):
    names = []
    name_ = self
    while name_.parent:
      names.insert(0, name_.name)
      name_ = name_.parent
    return "/".join(names) if names else ""

  def assign(self, call: CallEntry):
    assert not call.namespace
    call.namespace = self
    self.items.append(call)

  def _get_unique_name(self, suggested_name: str) -> str:
    if suggested_name not in self.childs_by_name:
      return suggested_name
    for i in itertools.count(1):
      suggested_name_ = f"{suggested_name}_{i}"
      if suggested_name_ not in self.childs_by_name:
        return suggested_name_

  def register(self, suggested_name: str, child: CallEntry) -> RegisteredName:
    assert not child.namespace
    name = self._get_unique_name(suggested_name)
    name_ = RegisteredName(parent=self, name=name)
    name_.assign(child)
    self.childs_by_name[name] = name_
    return name_

  def register_input(self, name: str, tensor: TensorEntry) -> RegisteredName:
    if name in self.childs_by_name:
      assert self.childs_by_name[name].tensor is tensor
      return self.childs_by_name[name]
    assert name not in self.childs_by_name
    name_ = RegisteredName(parent=self, name=name, tensor=tensor)
    self.childs_by_name[name] = name_
    return name_

  def name_for_tensor(self, tensor: TensorEntry) -> str:
    for name_ in tensor.names:
      if name_.parent is self:
        assert self.childs_by_name[name_.name] is name_
        return name_.name
    raise Exception(f"namespace {self!r}: tensor {tensor!r} not found")

  def dump(self, prefix=""):
    for name, child in self.childs_by_name.items():
      print(f"{prefix}{name}: {child.items}")
      child.dump(prefix=f"{prefix}  ")


class ReturnnContext:
  def __init__(self, *, parent: Optional[ReturnnContext] = None, name: Optional[str] = None):
    self.parent = parent
    if parent:
      self.config = parent.config
    else:
      self.config = Config({
        "debug_print_layer_output_template": True,
      })
    self.extern_data = ExternData()
    self.network = TFNetwork(
      extern_data=self.extern_data, config=self.config,
      parent_net=parent.network if parent else None,
      name="root" if not parent else "%s/%s" % (parent.network.name, name))


class Naming:
  tensors: WeakKeyDictionary[Tensor, TensorEntry]
  modules: OrderedDict[Module, ModuleEntry]
  inputs: List[Tensor]
  outputs: List[Tensor]
  module_creation_call_stack: List[ModuleEntry]
  func_call_stack: List[CallEntry]
  root_func_calls: List[CallEntry]
  _instance: Optional["Naming"] = None

  @classmethod
  def get_instance(cls) -> "Naming":
    if not cls._instance:
      cls._instance = Naming()
    return cls._instance

  def __init__(self):
    self.tensors = WeakKeyDictionary()
    self.modules = OrderedDict()
    self.inputs = []
    self.outputs = []
    self.module_creation_call_stack = []
    self.func_call_stack = []
    self.root_func_calls = []
    self.root_namespace = RegisteredName(parent=None)

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

  def pop_module_creation(self, module: Module):
    assert self.module_creation_call_stack[-1].module is module
    self.module_creation_call_stack.pop(-1)

  def push_func_call(self, *, module: Optional[Module] = None, func: Callable, inputs: List[Tensor]) -> CallEntry:
    module_entry = self.modules[module] if module else None
    entry = CallEntry(func=func, module=module_entry)
    entry.inputs = [self.tensors[x] for x in inputs]
    entry.level = len(self.func_call_stack)
    if self.func_call_stack:
      recent_entry = self.func_call_stack[-1]
      recent_entry.child_calls.append(entry)
      entry.parent_call = recent_entry
      if recent_entry.module and recent_entry.module.module.create_returnn_layer_dict:
        raise Exception(f"Not expected to have sub call stacks on module {recent_entry.module.module}")
    else:
      self.root_func_calls.append(entry)
    self.func_call_stack.append(entry)
    if entry.level == 0:
      if module_entry and not module_entry.parent_owning_modules:
        # Special case, somewhat nicer to flatten the namespace for this case.
        self.root_namespace.assign(entry)
      else:
        self.root_namespace.register(suggested_name=entry.get_canonical_name(), child=entry)
    else:
      assert entry.parent_call.namespace
      namespace = entry.parent_call.namespace
      namespace.register(suggested_name=entry.get_canonical_name(), child=entry)
    assert entry.namespace
    if module.create_returnn_layer_dict:
      assert entry.namespace.parent
      for x in entry.inputs:
        name = entry.namespace.parent.name_for_tensor(x)
        assert name
    return entry

  def pop_func_call(self, *, func: Callable, outputs: List[Tensor]):
    assert self.func_call_stack[-1].func is func
    entry_outputs = [self.register_tensor(x) for x in outputs]
    entry = self.func_call_stack.pop(-1)
    entry.outputs = entry_outputs
    for x in entry_outputs:
      x.output_from_calls.append(entry)
      if entry.module:
        x.output_from_modules.append(entry.module)
    if entry_outputs:
      entry_outputs[0].names.append(entry.namespace)
    if not entry.child_calls:
      assert entry.module
      assert entry.module.module.create_returnn_layer_dict

  def register_module_child_attr(self, parent: Module, attr: str, child: Union[Module, Tensor]):
    assert getattr(parent, attr) is child
    parent_entry = self.modules[parent]
    if child in self.modules:
      child_entry = self.modules[child]
      child_entry.parent_owning_modules.append(parent_entry)
    else:
      assert child in self.tensors
      child_entry = self.tensors[child]
      assert isinstance(child_entry, TensorEntry)
      for parent_param in parent.parameters(recurse=False):
        if parent_param is child:
          child_entry.is_param = True
          break
      child_entry.parent_owning_modules.append(parent_entry)

  def register_tensor(self, tensor: Tensor) -> TensorEntry:
    if tensor not in self.tensors:
      self.tensors[tensor] = TensorEntry(
        tensor=ref(tensor), creation_stack_call=self.func_call_stack[-1] if self.func_call_stack else None)
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
      child = root.register(suggested_name=call.get_canonical_name(), child=call)
      Naming._register_call_names(child, call.child_calls)

  def register_output(self, tensor: Tensor):
    assert tensor in self.tensors
    entry = self.tensors[tensor]
    assert isinstance(entry, TensorEntry)
    assert not entry.is_param and not entry.is_const and not entry.is_input  # not implemented, although simple...
    self.outputs.append(tensor)

    self.root_namespace.dump()
