
import sys
from typing import Any
import importlib
from ... import log
from ... import __package__ as _base_package
from ..context import WrapCtx


class WrappedObject:
  def __init__(self, orig_obj: Any, name: str, ctx: WrapCtx):
    self._wrapped__name = name
    self._wrapped__orig_obj = orig_obj
    self._wrapped__ctx = ctx

  def __repr__(self):
    if self._wrapped__orig_obj is self:
      return super(WrappedObject, self).__repr__()
    return "<WrappedObject %r type %s>" % (self._wrapped__name, type(self._wrapped__orig_obj))

  def __getattr__(self, item):
    if item == "_wrapped__orig_obj":
      # Special case. torch.Tensor functions would create new copies,
      # which would use our custom class, but then this is not initialized.
      return self
    if item == "_wrapped__name":
      return "<unknown>"
    if item == "_wrapped__ctx":
      return None
    if self._wrapped__orig_obj is self:  # special case
      raise AttributeError("No attrib %r" % item)
    from ..wrap import wrap
    res_ = getattr(self._wrapped__orig_obj, item)
    try:
      res = wrap(res_, name="%s.%s" % (self._wrapped__name, item), ctx=self._wrapped__ctx)
      if res is not res_:
        # Speedup, and avoid recreations.
        # Note that this can potentially be dangerous and lead to inconsistencies,
        # esp if the underlying object changes the value,
        # then this would not reflect that.
        # TODO maybe remove this?
        object.__setattr__(self, item, res)
      else:
        if log.Verbosity >= 10:
          log.unique_print("*** not wrapped: '%s.%s', type %r" % (self._wrapped__name, item, type(res)))
      if log.Verbosity >= 6:
        postfix = ""
        if type(res).__name__.startswith("Wrapped") or getattr(res, "__module__", "").startswith(_base_package):
          postfix = " -> %r" % res
        log.unique_print("*** indirect getattr '%s.%s'%s" % (self._wrapped__name, item, postfix))
    except AttributeError as exc:  # no exception expected. esp, we should **NOT** forward AttributeError
      raise RuntimeError(exc) from exc
    return res

  # setattr on this object directly, not on the wrapped one
  # Note again that this can potentially lead to inconsistencies. See note above.
  __setattr__ = object.__setattr__

  def __delattr__(self, item):
    if hasattr(self._wrapped__orig_obj, item):
      delattr(self._wrapped__orig_obj, item)
    if item in self.__dict__:
      del self.__dict__[item]

  def __dir__(self):
    if self._wrapped__orig_obj is self:
      return []
    return dir(self._wrapped__orig_obj)

  def __bool__(self):
    if self._wrapped__orig_obj is self:
      # noinspection PyUnresolvedReferences
      return super(WrappedObject, self).__bool__(self)
    return bool(self._wrapped__orig_obj)


def make_wrapped_object(obj, *, name: str, ctx: WrapCtx):
  # First check if we should also wrap the type.
  if type(obj) in ctx.explicit_wrapped_types:
    wrap_type = ctx.explicit_wrapped_types[type(obj)]
    return wrap_type.wrap(obj, name=name, ctx=ctx)

  if ctx.should_wrap_mod(getattr(type(obj), "__module__", "")) and type(obj) not in ctx.keep_as_is_types:
    from .class_ import WrappedClassBase
    assert getattr(sys.modules[obj.__class__.__module__], obj.__class__.__qualname__) is obj.__class__
    wrapped_mod = importlib.import_module(ctx.wrapped_mod_prefix + obj.__class__.__module__)
    cls = getattr(wrapped_mod, obj.__class__.__qualname__)
    assert issubclass(cls, WrappedClassBase)
    obj = cls(orig_obj=obj, name=name, ctx=ctx)
    return obj

  # Fallback, or standard case.
  return WrappedObject(obj, name=name, ctx=ctx)
