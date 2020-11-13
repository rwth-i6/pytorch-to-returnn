
import importlib
import importlib.abc
import importlib.machinery
import types
from typing import Optional
from .loader import MetaPathLoader


# https://docs.python.org/3/library/importlib.html
# https://www.python.org/dev/peps/pep-0302/
# https://dev.to/dangerontheranger/dependency-injection-with-import-hooks-in-python-3-5hap


class MetaPathFinder(importlib.abc.MetaPathFinder):
  def __init__(self, loader: MetaPathLoader):
    self.loader = loader
    self.ctx = loader.ctx

  def find_spec(self, fullname: str, path: Optional[str], target: Optional[types.ModuleType] = None):
    if fullname.startswith(self.loader.mod_prefix) or fullname == self.loader.mod_prefix[:-1]:
      return importlib.machinery.ModuleSpec(fullname, self.loader)
    return None
