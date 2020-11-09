"""
Utility to import a module with automatic Torch import wrapping, which replaces all::

  import torch

To::

  from pytorch_to_returnn.wrapped_torch as torch

In your user code, you would replace::

  import custom_torch_code

By::

  custom_torch_code = pytorch_to_returnn.wrapped_import.wrapped_import("custom_torch_code")

Both the wrapped and original module can be imported at the same time.
The wrapped module will internally get the full mod name ``pytorch_to_returnn._wrapped_mods.custom_torch_code``.
See :class:`_AstImportTransformer`.

"""

import torch
import ast
import types
import typing
from typing import Optional, Union, Set, Any
import sys
import os
import importlib
import importlib.abc
import importlib.machinery
import linecache


DEBUG = False


def _ast_get_source_segment(src_filename: str, node: ast.AST) -> Optional[str]:
    """Get source code segment of the *source* that generated *node*.

    If some location information (`lineno`, `end_lineno`, `col_offset`,
    or `end_col_offset`) is missing, return None.

    Original ast.get_source_segment is very inefficient. This is faster.
    """
    try:
        lineno = node.lineno - 1
        end_lineno = node.end_lineno - 1
        col_offset = node.col_offset
        end_col_offset = node.end_col_offset
    except AttributeError:
        return None

    lines = linecache.getlines(src_filename)
    if end_lineno == lineno:
        return lines[lineno].encode()[col_offset:end_col_offset].decode()

    first = lines[lineno].encode()[col_offset:].decode()
    last = lines[end_lineno].encode()[:end_col_offset].decode()
    lines = lines[lineno+1:end_lineno]

    lines.insert(0, first)
    lines.append(last)
    return ''.join(lines)


class _AstImportTransformer(ast.NodeTransformer):
  def __init__(self, base_mod_name: str, src_filename: str):
    self.src_filename = src_filename
    self.base_mod_name = base_mod_name  # e.g. "parallel_wavegan"
    pass

  def _should_wrap_mod_name(self, mod_name: str) -> bool:
    if mod_name == "torch" or mod_name.startswith("torch."):
      return True
    if mod_name == self.base_mod_name or mod_name.startswith(self.base_mod_name + "."):
      return True
    return False

  def visit_Import(self, node: ast.Import):
    # https://docs.python.org/3/library/ast.html#ast.Import
    if not any([self._should_wrap_mod_name(alias.name) for alias in node.names]):
      return node
    if len(node.names) >= 2:
      # Break down into multiple imports.
      res = []
      for alias in node.names:
        assert isinstance(alias, ast.alias)
        sub_node = ast.Import(names=[alias])
        ast.copy_location(sub_node, node)
        sub_node = self.visit_Import(sub_node)
        assert isinstance(sub_node, ast.Import)
        res.append(sub_node)
      return res
    if DEBUG:
      print("*** Import:", ast.dump(node))
      print("  ", _ast_get_source_segment(src_filename=self.src_filename, node=node))
    assert len(node.names) == 1
    alias, = node.names
    assert isinstance(alias, ast.alias)
    if not self._should_wrap_mod_name(alias.name):
      return node
    # E.g. `import parallel_wavegan.layers`
    # ->
    # `import pytorch_to_returnn._wrapped_mods.parallel_wavegan.layers`
    # If "." in mod name, also:
    # `from pytorch_to_returnn._wrapped_mods import parallel_wavegan`
    # If `import parallel_wavegan.layers as x`, simpler:
    if alias.asname:  # import torch as x -> import pytorch_to_returnn._wrapped_mods.torch as x
      new_node = ast.Import(
        names=[ast.alias(
          name=_ModPrefix + alias.name, asname=alias.asname)])
      ast.copy_location(new_node, node)
      return new_node
    # If "." in mod name, simpler:
    if "." not in alias.name:  # import torch -> import pytorch_to_returnn._wrapped_mods.torch as torch
      new_node = ast.Import(
        names=[ast.alias(
          name=_ModPrefix + alias.name, asname=alias.name)])
      ast.copy_location(new_node, node)
      return new_node
    assert not alias.asname
    new_import_node = ast.Import(names=[ast.alias(name=_ModPrefix + alias.name)])
    ast.copy_location(new_import_node, node)
    assert "." in alias.name
    base_mod_name = alias.name.partition(".")[0]
    new_alias_node = ast.ImportFrom(  # from pytorch_to_returnn._wrapped_mods import torch
      module=_ModPrefix[:-1], names=[ast.alias(name=base_mod_name)])
    ast.copy_location(new_alias_node, node)
    return [new_import_node, new_alias_node]

  def visit_ImportFrom(self, node: ast.ImportFrom):
    # https://docs.python.org/3/library/ast.html#ast.ImportFrom
    if node.level == 0 and not self._should_wrap_mod_name(node.module):
      return node
    if DEBUG:
      print("*** ImportFrom:", ast.dump(node))
      print("  ", _ast_get_source_segment(src_filename=self.src_filename, node=node))
    if node.level > 0:
      return node  # relative imports should already be handled correctly
    assert node.level == 0
    new_node = ast.ImportFrom(
      module=_ModPrefix + node.module, names=node.names, level=0)
    ast.copy_location(new_node, node)
    return new_node


class WrappedModule(types.ModuleType):
  def __init__(self, *, name: str, orig_mod: types.ModuleType):
    super(WrappedModule, self).__init__(name=name)
    self._wrapped__orig_mod = orig_mod
    assert not isinstance(orig_mod, WrappedModule)


class WrappedSourceModule(WrappedModule):
  def __init__(self, *, source: str, **kwargs):
    super(WrappedSourceModule, self).__init__(**kwargs)
    self._wrapped__source = source


_ExplicitIndirectModList = {
  "torch",  # leave this in to not do any magic on the original Torch modules
  "torch.tensor", "torch.distributed"
}


class WrappedIndirectModule(WrappedModule):
  """
  This can not be transformed on source level,
  so we wrap all attrib access on-the-fly.
  """
  def __init__(self, **kwargs):
    super(WrappedIndirectModule, self).__init__(**kwargs)
    if DEBUG:
      print("*** WrappedIndirectModule", self.__name__)
    if getattr(self._wrapped__orig_mod, "__all__", None) is not None:
      self.__all__ = self._wrapped__orig_mod.__all__
    else:
      names = sorted(vars(self._wrapped__orig_mod).keys())
      self.__all__ = [name for name in names if name[:1] != "_"]

  def __getattr__(self, item):
    assert item not in {"_wrapped__orig_mod", "__name__"}
    print("*** WrappedIndirectModule %s getattr %r" % (self.__name__, item))
    return getattr(self._wrapped__orig_mod, item)

  def __setattr__(self, key, value):
    if key in {"_wrapped__orig_mod", "__all__", "__loader__", "__package__", "__spec__", "__name__", "__path__"}:
      return super(WrappedIndirectModule, self).__setattr__(key, value)
    print("*** WrappedIndirectModule %s setattr %r, %r" % (self.__name__, key, value))
    return setattr(self._wrapped__orig_mod, key, value)


class WrappedTensor:
  pass


# https://docs.python.org/3/library/importlib.html
# https://www.python.org/dev/peps/pep-0302/
# https://dev.to/dangerontheranger/dependency-injection-with-import-hooks-in-python-3-5hap


_ModPrefix = "%s._wrapped_mods." % __package__


class _MetaPathLoader(importlib.abc.Loader):
  def __repr__(self):
    return "<wrapped mod loader>"

  def create_module(self, spec: importlib.machinery.ModuleSpec) -> WrappedModule:
    assert spec.name.startswith(_ModPrefix)
    orig_mod_name = spec.name[len(_ModPrefix):]
    # Normal load. This is just to get __file__, and check whether it is correct. Maybe useful otherwise as well.
    orig_mod = importlib.import_module(orig_mod_name)
    assert orig_mod.__name__ == orig_mod_name
    orig_mod_name_parts = orig_mod_name.split(".")
    for i in range(1, len(orig_mod_name_parts) + 1):
      if ".".join(orig_mod_name_parts[:i]) in _ExplicitIndirectModList:
        return WrappedIndirectModule(name=spec.name, orig_mod=orig_mod)
    orig_mod_loader = orig_mod.__loader__
    if not isinstance(orig_mod_loader, importlib.abc.ExecutionLoader):
      return WrappedIndirectModule(name=spec.name, orig_mod=orig_mod)
    src = orig_mod_loader.get_source(orig_mod.__name__)
    if src is None:  # e.g. binary module
      return WrappedIndirectModule(name=spec.name, orig_mod=orig_mod)
    return WrappedSourceModule(name=spec.name, orig_mod=orig_mod, source=src)

  def exec_module(self, module: WrappedModule):
    if DEBUG:
      print("*** exec mod", module)
    assert isinstance(module, WrappedModule)
    assert module.__name__.startswith(_ModPrefix)
    if isinstance(module, WrappedIndirectModule):
      return  # nothing needed to be done
    assert isinstance(module, WrappedSourceModule)
    # noinspection PyProtectedMember
    orig_mod = module._wrapped__orig_mod
    # noinspection PyProtectedMember
    src = module._wrapped__source
    tree = ast.parse(source=src, filename=orig_mod.__file__)
    ast_transformer = _AstImportTransformer(
      base_mod_name=orig_mod.__name__.partition(".")[0],
      src_filename=orig_mod.__file__)
    tree = ast_transformer.visit(tree)
    tree = ast.fix_missing_locations(tree)
    code = compile(tree, orig_mod.__file__, "exec")
    is_pkg = getattr(orig_mod, "__path__", None) is not None
    module.__file__ = orig_mod.__file__
    module.__loader__ = self
    if is_pkg:
        module.__path__ = []
        module.__package__ = module.__name__
    else:
        module.__package__ = module.__name__.rpartition('.')[0]
    exec(code, module.__dict__)


class _MetaPathFinder(importlib.abc.MetaPathFinder):
  def __init__(self, loader: _MetaPathLoader):
    self._loader = loader

  def find_spec(self, fullname: str, path: Optional[str], target: Optional[types.ModuleType] = None):
    if fullname.startswith(_ModPrefix):
      return importlib.machinery.ModuleSpec(fullname, self._loader)
    return None


_loader = _MetaPathLoader()
_finder = _MetaPathFinder(loader=_loader)


def wrapped_import(mod_name: str) -> Union[WrappedModule, Any]:  # rtype Any to make type checkers happy
  """
  :param str mod_name: full mod name, e.g. "custom_torch_code"
  :return: wrapped module
  """
  if _finder not in sys.meta_path:
    sys.meta_path.append(_finder)
  mod = importlib.import_module(_ModPrefix + mod_name)
  assert isinstance(mod, WrappedModule)
  return mod


if typing.TYPE_CHECKING:
  # This also currently does not work...
  # https://youtrack.jetbrains.com/issue/PY-45376
  wrapped_import = importlib.import_module
