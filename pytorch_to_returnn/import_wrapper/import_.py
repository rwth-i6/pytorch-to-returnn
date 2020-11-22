
import sys
import types
from typing import Union, Any
import importlib
from .base_wrappers import WrappedModule
from .context import WrapCtx
from .meta_path import MetaPathFinder, MetaPathLoader
from .. import log


WrappedModPrefixes = set()


def import_module(mod_name: str, *, ctx: WrapCtx) -> Union[types.ModuleType, WrappedModule, Any]:
  _maybe_register_meta_path(ctx)
  if ctx.mod_map.should_wrap_mod_name(mod_name):
    mapped_mod_name = ctx.mod_map.map_mod_name(mod_name)
  else:
    mapped_mod_name = ctx.wrapped_mod_prefix + mod_name
  mod = importlib.import_module(mapped_mod_name)
  if mapped_mod_name.startswith(ctx.wrapped_mod_prefix):
    assert isinstance(mod, WrappedModule)
  assert isinstance(mod, types.ModuleType)
  return mod


def _maybe_register_meta_path(ctx: WrapCtx):
  for meta_path in sys.meta_path:
    if isinstance(meta_path, MetaPathFinder):
      if meta_path.ctx is ctx:  # TODO?
        return

  if log.Verbosity >= 4:
    print(f"*** register sys.meta_path for ctx {ctx}")
  loader = MetaPathLoader(ctx=ctx)
  finder = MetaPathFinder(loader=loader)
  sys.meta_path.append(finder)
  WrappedModPrefixes.add(ctx.wrapped_mod_prefix)
