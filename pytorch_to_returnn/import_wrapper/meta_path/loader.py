
import importlib
import importlib.abc
import importlib.machinery
import sys
import ast
from ... import log
from ..ast_transformer import AstImportTransformer
from ..base_wrappers.module import WrappedModule, WrappedSourceModule, WrappedIndirectModule
from ..context import WrapCtx


# https://docs.python.org/3/library/importlib.html
# https://www.python.org/dev/peps/pep-0302/
# https://dev.to/dangerontheranger/dependency-injection-with-import-hooks-in-python-3-5hap


class MetaPathLoader(importlib.abc.Loader):
  def __init__(self, ctx: WrapCtx):
    self.ctx = ctx
    self.mod_prefix = ctx.wrapped_mod_prefix  # e.g. "pytorch_to_returnn._wrapped_mods."
    assert self.mod_prefix.endswith(".")

  def __repr__(self):
    return "<wrapped mod loader>"

  def create_module(self, spec: importlib.machinery.ModuleSpec) -> WrappedModule:
    if spec.name == self.mod_prefix[:-1]:
      return None  # fall back to default
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
      if ".".join(orig_mod_name_parts[:i]) in self.ctx.wrap_mods_direct:
        explicit_direct_use = True
        break
      if ".".join(orig_mod_name_parts[:i]) in self.ctx.wrap_mods_indirect:
        return WrappedIndirectModule(name=spec.name, orig_mod=orig_mod, ctx=self.ctx)
    orig_mod_loader = orig_mod.__loader__
    if not isinstance(orig_mod_loader, importlib.abc.ExecutionLoader):
      assert not explicit_direct_use
      return WrappedIndirectModule(name=spec.name, orig_mod=orig_mod, ctx=self.ctx)
    src = orig_mod_loader.get_source(orig_mod.__name__)
    if src is None:  # e.g. binary module
      assert not explicit_direct_use
      return WrappedIndirectModule(name=spec.name, orig_mod=orig_mod, ctx=self.ctx)
    return WrappedSourceModule(name=spec.name, orig_mod=orig_mod, source=src, ctx=self.ctx)

  def exec_module(self, module: WrappedModule):
    if module.__name__ == self.mod_prefix[:-1]:  # dummy top-level module
      module.__file__ = None
      module.__loader__ = self
      module.__path__ = []
      module.__package__ = module.__name__
      return
    assert isinstance(module, WrappedModule)
    assert module.__name__.startswith(self.mod_prefix)
    if isinstance(module, WrappedIndirectModule):
      return  # nothing needed to be done
    if log.Verbosity >= 5:
      print("*** exec mod", module)
    assert isinstance(module, WrappedSourceModule)
    # noinspection PyProtectedMember
    orig_mod = module._wrapped__orig_mod
    orig_mod_name = module.__name__[len(self.mod_prefix):]  # e.g. "parallel_wavegan.layers"
    assert orig_mod.__name__ == orig_mod_name
    if not self.ctx.should_wrap_mod(orig_mod_name):
      if log.Verbosity >= 4:
        print(f"*** {self.ctx} extend by mod {orig_mod_name!r}")
      self.ctx.extend_wrap_mod_(orig_mod_name)
    # noinspection PyProtectedMember
    src = module._wrapped__source
    tree = ast.parse(source=src, filename=orig_mod.__file__)
    ast_transformer = AstImportTransformer(
      mod_map=self.ctx.mod_map,
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
