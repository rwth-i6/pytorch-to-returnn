
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
  creation_stack_call: Optional[CallEntry]

  def __init__(self, tensor: ref[Tensor], creation_stack_call: Optional[CallEntry]):
    self.tensor = tensor
    self.creation_stack_call = creation_stack_call
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
  level: int = 0
  items: List[CallEntry]  # can be multiple merged together

  def __init__(self, *, parent: Optional["RegisteredName"]):
    self.childs_by_name = OrderedDict()
    self.parent = parent
    self.items = []
    if parent:
      self.level = parent.level + 1

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
    name_ = RegisteredName(parent=self)
    name_.assign(child)
    self.childs_by_name[name] = name_
    return name_

  def dump(self, prefix=""):
    for name, child in self.childs_by_name.items():
      print(f"{prefix}{name}: {child.items}")
      child.dump(prefix=f"{prefix}  ")


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

  def push_func_call(self, *, module: Optional[Module] = None, func: Callable, inputs: List[Tensor]):
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

  def pop_func_call(self, *, func: Callable, outputs: List[Tensor]):
    assert self.func_call_stack[-1].func is func
    entry = self.func_call_stack.pop(-1)
    entry.outputs = []
    for x in outputs:
      x = self.register_tensor(x)
      x.output_from_calls.append(entry)
      if entry.module:
        x.output_from_modules.append(entry.module)
      entry.outputs.append(x)

  def _filter_tensor_inputs(self, inputs):
    # TODO other tensor types? generic enough?
    from .torch import Tensor
    return [x for x in inputs if isinstance(x, Tensor)]

  @staticmethod
  def wrap_func(func: Callable) -> Callable:
    def wrapped_func(*inputs):
      self = Naming.get_instance()
      self.push_func_call(func=func, inputs=self._filter_tensor_inputs(inputs))
      output = func(*inputs)
      if isinstance(output, (list, tuple)):
        outputs = list(output)
      else:
        outputs = [output]
      self.pop_func_call(func=func, outputs=outputs)
      return output
    wrapped_func.__name__ = func.__name__
    wrapped_func.__qualname__ = func.__qualname__
    wrapped_func.__module__ = func.__module__
    return wrapped_func

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

  @staticmethod
  def _register_call_names(root: RegisteredName, calls: List[CallEntry]):
    for call in calls:
      child = root.register(suggested_name=call.get_canonical_name(), child=call)
      Naming._register_call_names(child, call.child_calls)

  def register_output(self, tensor: Tensor):
    assert tensor in self.tensors
    entry = self.tensors[tensor]
    assert isinstance(entry, TensorEntry)
    assert not entry.is_param and not entry.is_const and not entry.is_input
    self.outputs.append(tensor)

    def _get_calls(entry_: TensorEntry):
      assert entry_.output_from_calls
      res = []
      for call_ in entry_.output_from_calls:
        assert isinstance(call_, CallEntry)
        assert entry_ in call_.outputs
        if call_.level > 0:  # only consider top-level calls
          continue
        res.append(call_)
      return res

    def _get_childs(call_: CallEntry):
      res = []
      for entry_ in call_.inputs:
        if entry_.is_input:
          continue
        res.extend(_get_calls(entry_))
      return res

    calls = list(_breadth_first_search(seed=_get_calls(entry), childs=_get_childs))
    calls.reverse()
    assert calls
    for call in calls:
      if call.module and not call.module.parent_owning_modules:  # special case, flatten this away
        self.root_namespace.assign(call)
        Naming._register_call_names(self.root_namespace, call.child_calls)
      else:
        child = self.root_namespace.register(suggested_name=call.get_canonical_name(), child=call)
        Naming._register_call_names(child, call.child_calls)

    self.root_namespace.dump()

    # TODO get level 1 call
    # TODO now get path module -> tensor
