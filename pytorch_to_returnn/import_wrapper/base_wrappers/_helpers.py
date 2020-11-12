
from ..context import WrapCtx


def copy_attribs_qualname_and_co(target, source, *, ctx: WrapCtx):
  target.__name__ = source.__name__
  target.__qualname__ = source.__qualname__
  if source.__module__:
    if ctx.should_wrap_mod(source.__module__):
      target.__module__ = ctx.mod_map.map_mod_name(source.__module__)
    else:
      target.__module__ = source.__module__
