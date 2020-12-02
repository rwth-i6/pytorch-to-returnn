
from __future__ import annotations
from typing import Optional, List, Tuple
from . import _types
from . import call as _call
from . import namespace as _namespace
from . import naming as _naming


class ModuleEntry:
  module: _types.Module
  level: Optional[int] = None
  calls: List[_call.CallEntry]
  names: List[_namespace.RegisteredName]
  canonical_name: Optional[str] = None
  parent_owning_modules: List[Tuple[ModuleEntry, str]]
  parent_context_modules: List[ModuleEntry]

  def __init__(self, module: _types.Module):
    self.module = module
    self.calls = []
    self.names = []
    self.parent_owning_modules = []
    self.parent_context_modules = []

  def __repr__(self):
    module_repr = repr(self.module)
    # torch.nn.Module.__repr__ can be too verbose when there are children...
    # Strip that away.
    if "\n" in module_repr:
      lines = module_repr.splitlines()
      assert len(lines) >= 2
      module_repr = f"{lines[0].strip()}...{lines[-1].strip()}"
    return f"<ModuleEntry {module_repr}>"

  def get_parent_calling_modules(self) -> List[ModuleEntry]:
    res = []
    for call in self.calls:
      assert call.module is self
      while call.parent_call:
        call = call.parent_call
        if call.module:
          res.append(call.module)
          break
    return res

  def get_root_owning_module(self) -> ModuleEntry:
    mod = self
    while mod.parent_owning_modules:
      mod = mod.parent_owning_modules[0][0]
    return mod

  def get_canonical_name(self, parent_namespace: Optional[_namespace.RegisteredName] = None, *, _visited=None) -> str:
    if self.canonical_name:
      return self.canonical_name
    if _visited is None:
      _visited = set()
    _visited.add(self)
    naming = _naming.Naming.get_instance()
    if parent_namespace is None:
      parent_namespace = naming.root_namespace
    if self.parent_owning_modules:
      mod, name = self.parent_owning_modules[0]
      if parent_namespace and mod in parent_namespace.modules:
        prefix = ""
      elif not mod.module.has_torch_forward() and mod not in _visited:
        prefix = mod.get_canonical_name(_visited=_visited)
        if prefix:
          prefix += "_"
      else:
        prefix = ""
      if not prefix and name[:1].isnumeric():
        return f"layer{name}"
      return prefix + name
    if parent_namespace and self in parent_namespace.modules:
      return self.module.get_returnn_name()
    if set(self.parent_context_modules).intersection(_visited):
      return self.module.get_returnn_name()
    if parent_namespace and parent_namespace is not naming.root_namespace:
      for mod in self.parent_context_modules:
        if mod in parent_namespace.modules:
          return self.module.get_returnn_name()
    prefix = ""
    for mod in reversed(self.parent_context_modules):
      prefix = mod.get_canonical_name(_visited=_visited, parent_namespace=parent_namespace)
      if prefix:
        prefix += "_"
        break
    return prefix + self.module.get_returnn_name()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    if not exc_type:
      _naming.Naming.get_instance().pop_module_context(self.module)

