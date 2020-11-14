
from __future__ import annotations

import typing
from typing import Optional, Any, List, TypeVar, Dict, Callable, Iterable, Union
import weakref
from weakref import WeakKeyDictionary, ref
from collections import OrderedDict
import itertools

if typing.TYPE_CHECKING:
  # Just for typing. Although we also cover traced/wrapped Torch.
  from .torch import Tensor
  from .torch.nn import Module


class TensorEntry:
  tensor: ref[Tensor]
  is_param: bool = False
  is_const: bool = False  # e.g. via from_numpy, empty, zeros, etc
  is_input: bool = False  # in TF1 terminology, would be a placeholder
  name: Optional[str] = None
  output_from_modules: List["ModuleEntry"]
  output_from_calls: List["CallEntry"]
  parent_owning_modules: List["ModuleEntry"]  # e.g. param or buffer

  def __init__(self, tensor: ref[Tensor]):
    self.tensor = tensor
    self.output_from_modules = []
    self.output_from_calls = []
    self.parent_owning_modules = []


class CallEntry:
  """
  Can be a module() call, or regular func.
  Note that a module can be called multiple times.
  """
  func: Callable
  module: Optional["ModuleEntry"]
  inputs: Optional[List["TensorEntry"]]
  outputs: Optional[List["TensorEntry"]]
  parent_call: Optional["CallEntry"] = None  # parent in the call stack
  child_calls: List["CallEntry"]
  level: Optional[int] = None

  def __init__(self, func: Callable, module: Optional["ModuleEntry"]):
    self.func = func
    self.module = module
    self.child_calls = []

  def get_root_call(self) -> "CallEntry":
    entry = self
    while entry.parent_call:
      entry = entry.parent_call
    return entry


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


_SearchType = TypeVar("_SearchType")
_SearchChildsFuncT = Callable[[_SearchType], Iterable[_SearchType]]
_SearchFilterFuncT = Callable[[_SearchType], bool]


def _breadth_first_search(
      root: _SearchType, childs: _SearchChildsFuncT):
  visited = set()
  queue = [root]
  while queue:
    next_queue = []
    for elem in queue:
      if elem in visited:
        continue
      visited.add(elem)
      yield elem
      next_queue.extend(childs(elem))
    queue = next_queue


def _breadth_first_search_first(
      root: _SearchType, childs: _SearchChildsFuncT, filter_elem: Optional[_SearchFilterFuncT] = None):
  for elem in _breadth_first_search(root=root, childs=childs):
    if filter_elem and not filter_elem(elem):
      continue
    return elem
  return None


class RegisteredName:
  childs_by_name: OrderedDict[str, "RegisteredName"]
  parent: Optional["RegisteredName"]
  level: int = 0
  item: Optional[CallEntry]

  def __init__(self, *, parent: Optional["RegisteredName"], item: Optional[CallEntry]):
    self.childs_by_name = OrderedDict()
    self.parent = parent
    self.item = item
    if parent:
      self.level = parent.level + 1

  def _get_unique_name(self, suggested_name: str) -> str:
    if suggested_name not in self.childs_by_name:
      return suggested_name
    for i in itertools.count(1):
      suggested_name_ = f"{suggested_name}_{i}"
      if suggested_name_ not in self.childs_by_name:
        return suggested_name_

  def register(self, suggested_name: str, child: CallEntry):
    name = self._get_unique_name(suggested_name)
    self.childs_by_name[name] = RegisteredName(parent=self, item=child)


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
    self.root_namespace = RegisteredName(parent=None, item=None)

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

  def push_func_call(self, *, module: Optional[Module] = None, func: Optional[Any] = None, inputs: List[Tensor]):
    module_entry = self.modules[module] if module else None
    entry = CallEntry(func=func, module=module_entry)
    entry.inputs = [self.tensors[x] for x in inputs]
    entry.level = len(self.func_call_stack)
    if self.func_call_stack:
      recent_entry = self.func_call_stack[-1]
      recent_entry.child_calls.append(entry)
      entry.parent_call = recent_entry
    else:
      self.root_func_calls.append(entry)
    self.func_call_stack.append(entry)
    return entry

  def pop_func_call(self, *, func: Optional[Any] = None, outputs: List[Tensor]):
    assert self.func_call_stack[-1].func is func
    entry = self.func_call_stack.pop(-1)
    entry.outputs = []
    for x in outputs:
      x = self.register_tensor(x)
      x.output_from_calls.append(entry)
      if entry.module:
        x.output_from_modules.append(entry.module)
      entry.outputs.append(x)

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
      self.tensors[tensor] = TensorEntry(tensor=ref(tensor))
    return self.tensors[tensor]

  def register_input(self, tensor: Tensor):
    entry = self.register_tensor(tensor)
    entry.is_input = True
    self.inputs.append(tensor)

  def register_output(self, tensor: Tensor):
    assert tensor in self.tensors
    entry = self.tensors[tensor]
    assert isinstance(entry, TensorEntry)
    assert not entry.is_param and not entry.is_const and not entry.is_input
    self.outputs.append(tensor)

    def _get_childs(entry_: TensorEntry):
      assert entry_.output_from_calls
      res = []
      for call_ in entry_.output_from_calls:
        assert isinstance(call_, CallEntry)
        assert entry_ in call_.outputs
        if call_.level > 0:  # only consider top-level calls
          continue
        if call_.module:  # not recurse, consider this below
          continue
        res.extend(call_.inputs)
      return res

    def _get_tensor_output_module(entry_: TensorEntry) -> Optional[ModuleEntry]:
      for call_ in entry_.output_from_calls:
        assert isinstance(call_, CallEntry)
        assert entry_ in call_.outputs
        if call_.level > 0:  # only consider top-level calls
          continue
        if call_.module:
          return call_.module
      return None

    def _is_tensor_output_from_module(entry_: TensorEntry) -> bool:
      return bool(_get_tensor_output_module(entry_))

    mod_t_entry = _breadth_first_search_first(entry, childs=_get_childs, filter_elem=_is_tensor_output_from_module)
    if not mod_t_entry:
      # No modules found. What should we do?
      return  # TODO
    assert isinstance(mod_t_entry, TensorEntry)
    module = _get_tensor_output_module(mod_t_entry)
    assert module
    module = module.get_root_owning_module()
