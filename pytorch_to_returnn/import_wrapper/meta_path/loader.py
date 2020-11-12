
import importlib
import importlib.abc
import importlib.machinery
import sys
import ast
from ... import log
from ..ast_transformer import AstImportTransformer


class MetaPathLoader(importlib.abc.Loader):
  def __init__(self, mod_prefix: str):
    """
    :param mod_prefix: e.g. "pytorch_to_returnn._wrapped_mods."
    """
    assert mod_prefix.endswith(".")
    self.mod_prefix = mod_prefix

  def __repr__(self):
    return "<wrapped mod loader>"

  def create_module(self, spec: importlib.machinery.ModuleSpec) -> WrappedModule:
    assert spec.name.startswith(self.mod_prefix)
    assert spec.name not in sys.modules
    orig_mod_name = spec.name[len(self.mod_prefix):]
    assert not orig_mod_name.startswith(self.mod_prefix)
    # Normal load. This is just to get __file__, and check whether it is correct. Maybe useful otherwise as well.
    orig_mod = importlib.import_module(orig_mod_name)
    assert not isinstance(orig_mod, WrappedModule)
    assert orig_mod.__name__ == orig_mod_name
    orig_mod_name_parts = orig_mod_name.split(".")
    explicit_direct_use = False
    for i in reversed(range(1, len(orig_mod_name_parts) + 1)):
      if ".".join(orig_mod_name_parts[:i]) in _ExplicitDirectModList:
        explicit_direct_use = True
        break
      if ".".join(orig_mod_name_parts[:i]) in _ExplicitIndirectModList:
        return WrappedIndirectModule(name=spec.name, orig_mod=orig_mod)
    orig_mod_loader = orig_mod.__loader__
    if not isinstance(orig_mod_loader, importlib.abc.ExecutionLoader):
      assert not explicit_direct_use
      return WrappedIndirectModule(name=spec.name, orig_mod=orig_mod)
    src = orig_mod_loader.get_source(orig_mod.__name__)
    if src is None:  # e.g. binary module
      assert not explicit_direct_use
      return WrappedIndirectModule(name=spec.name, orig_mod=orig_mod)
    return WrappedSourceModule(name=spec.name, orig_mod=orig_mod, source=src)

  def exec_module(self, module: WrappedModule):
    assert isinstance(module, WrappedModule)
    assert module.__name__.startswith(self.mod_prefix)
    if isinstance(module, WrappedIndirectModule):
      return  # nothing needed to be done
    if log.Verbosity >= 5:
      print("*** exec mod", module)
    assert isinstance(module, WrappedSourceModule)
    # noinspection PyProtectedMember
    orig_mod = module._wrapped__orig_mod
    # noinspection PyProtectedMember
    src = module._wrapped__source
    tree = ast.parse(source=src, filename=orig_mod.__file__)
    ast_transformer = AstImportTransformer(
      base_mod_names={orig_mod.__name__.partition(".")[0], "torch"},
      new_import_prefix=self.mod_prefix,
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
