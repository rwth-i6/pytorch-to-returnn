
import types
from ... import log
from .object import WrappedObject


def make_wrapped_function(func, name: str):
  def _call(*args, **kwargs):
    if log.Verbosity >= 3:
      log.unique_print("*** func call %s(...)" % name)
    res = func(*unwrap(args), **unwrap(kwargs))
    res = wrap(res, name="%s(...)" % name)
    return res

  _call.__name__ = func.__name__
  _call.__qualname__ = func.__qualname__
  if func.__module__:
    if _should_wrap_mod(func.__module__):
      _call.__module__ = _ModPrefix + func.__module__
    else:
      _call.__module__ = func.__module__

  if isinstance(func, types.FunctionType):
    return _call

  # This is maybe slightly unconventional / non-straightforward.
  # We construct it such to make PyTorch _add_docstr happy,
  # which expects certain types.
  class WrappedFunc(WrappedObject):
    def __new__(cls, *args, **kwargs):
      if log.Verbosity >= 3:
        log.unique_print("*** func call %s(...)" % name)
      return _call(*args, **kwargs)

  WrappedFunc.__name__ = func.__name__
  WrappedFunc.__qualname__ = func.__qualname__
  if func.__module__:
    if _should_wrap_mod(func.__module__):
      WrappedFunc.__module__ = _ModPrefix + func.__module__
    else:
      WrappedFunc.__module__ = func.__module__
  return WrappedFunc
