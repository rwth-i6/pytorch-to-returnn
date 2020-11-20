
import types
from ... import log
from .object import WrappedObject
from .utils import copy_attribs_qualname_and_co
from ..context import WrapCtx


def make_wrapped_function(func, name: str, ctx: WrapCtx):
  from ..wrap import wrap, unwrap

  def _call(*args, **kwargs):
    if log.Verbosity >= 3:
      log.unique_print("*** func call %s(...)" % name)
    res = func(*unwrap(args), **unwrap(kwargs))
    res = wrap(res, name="%s(...)" % name, ctx=ctx)
    return res

  copy_attribs_qualname_and_co(_call, func, ctx=ctx)
  if isinstance(func, _FuncType):
    return _call

  # This is maybe slightly unconventional / non-straightforward.
  # We construct it such to make PyTorch _add_docstr happy,
  # which expects certain types.
  class WrappedFunc(WrappedObject):
    def __new__(cls, *args, **kwargs):
      if log.Verbosity >= 3:
        log.unique_print("*** func call %s(...)" % name)
      return _call(*args, **kwargs)

  copy_attribs_qualname_and_co(WrappedFunc, func, ctx=ctx)
  return WrappedFunc


# https://youtrack.jetbrains.com/issue/PY-45382
# noinspection PyTypeChecker
_FuncType = types.FunctionType  # type: type
