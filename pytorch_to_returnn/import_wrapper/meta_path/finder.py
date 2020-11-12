
import importlib
import importlib.abc
import importlib.machinery
import types
from typing import Optional
from .loader import MetaPathLoader


class MetaPathFinder(importlib.abc.MetaPathFinder):
  def __init__(self, loader: MetaPathLoader):
    self._loader = loader

  def find_spec(self, fullname: str, path: Optional[str], target: Optional[types.ModuleType] = None):
    if fullname.startswith(self._loader.mod_prefix):
      return importlib.machinery.ModuleSpec(fullname, self._loader)
    return None
