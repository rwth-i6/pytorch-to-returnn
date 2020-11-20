
from ... import log
from .object import WrappedObject
from ..context import WrapCtx
from .utils import copy_attribs_qualname_and_co


class WrappedClassBase(WrappedObject):
  pass


def make_wrapped_class(cls: type, name: str, ctx: WrapCtx):
  from ..wrap import wrap, unwrap

  class WrappedClass(WrappedClassBase):
    def __init__(self, *args, **kwargs):
      if log.Verbosity >= 4:
        log.unique_print("*** WrappedClass %s(...)" % (name,))
      if isinstance(kwargs.get("orig_obj"), cls):
        assert not args
        WrappedObject.__init__(self, **kwargs)
      else:
        obj = cls(*unwrap(args), **unwrap(kwargs))
        WrappedObject.__init__(self, orig_obj=obj, name="%s(...)" % name, ctx=ctx)

    # Use __getattribute__ because we might have added `cls` as a base,
    # and want to catch all getattr, also to existing methods.
    def __getattribute__(self, item):
      # if item == "__dict__":
        # Special case. Directly get it from wrapped object.  # TODO use this?
        #  return self._wrapped__orig_obj.__dict__
      if item in {"_wrapped__orig_obj", "_wrapped__name", "_wrapped__ctx", "__class__"}:  # extra checks, and fast path
        return object.__getattribute__(self, item)
      if self._wrapped__orig_obj is self:  # special case
        res = object.__getattribute__(self, item)
        res = wrap(res, name="%s.%s" % (self._wrapped__name, item), ctx=ctx)
        return res
      return object.__getattribute__(self, item)

    def __setattr__(self, key, value):
      if key in {"_wrapped__orig_obj", "_wrapped__name", "_wrapped__ctx"}:
        return object.__setattr__(self, key, value)
      if self is self._wrapped__orig_obj:  # special case
        return object.__setattr__(self, key, value)
      return setattr(self._wrapped__orig_obj, key, value)

  copy_attribs_qualname_and_co(WrappedClass, cls, ctx=ctx)
  try:
    # We add this such that `isinstance(WrappedClass(), cls)` returns True.
    # Note that this could cause some problems (that's why we use __getattribute__ above),
    # and also, it does not always work (e.g. for native types like torch.Tensor).
    WrappedClass.__bases__ += (cls,)
  except TypeError:  # e.g. object layout differs or so
    pass  # just ignore
  return WrappedClass
