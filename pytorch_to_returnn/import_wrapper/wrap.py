
from collections import OrderedDict, Counter
import types
from typing import Tuple
import importlib
import sys
from .context import WrapCtx
from .base_wrappers import WrappedObject, WrappedModule, WrappedSourceModule
from .base_wrappers import make_wrapped_object, make_wrapped_class, make_wrapped_function


def wrap(obj, *, name: str, ctx: WrapCtx):
  if isinstance(obj, (WrappedObject, WrappedModule)):
    return obj
  if isinstance(obj, ctx.keep_as_is_types):
    return obj
  obj = _nested_transform(obj, lambda _x: wrap(_x, name="%s..." % name, ctx=ctx))

  if isinstance(obj, types.ModuleType):
    if ctx.should_wrap_mod(obj.__name__):
      if obj.__name__ not in sys.modules:
        # If this happens, this is actually a bug in how the module importing is done.
        # This is not on our side.
        # This might happen for native Torch modules.
        # This is slightly problematic for our following logic, because import_module will not find it.
        # So we patch this here.
        sys.modules[obj.__name__] = obj
      # If this comes from an WrappedIndirectModule, this will create a new WrappedIndirectModule for the sub mod.
      # See logic in _MetaPathLoader.
      obj = importlib.import_module(ctx.wrapped_mod_prefix + obj.__name__)
  elif isinstance(obj, _FuncTypes):
    if not obj.__module__ or ctx.should_wrap_mod(obj.__module__):
      obj = make_wrapped_function(func=obj, name=name, ctx=ctx)
  elif isinstance(obj, type):
    if obj in ctx.explicit_wrapped_types:
      obj = ctx.explicit_wrapped_types[obj].new_type
    elif obj in ctx.keep_as_is_types:
      pass  # keep as-is
    elif type(obj) != type:  # explicitly do not check sub-types, e.g. pybind11_type
      pass
    elif obj.__module__ and ctx.should_wrap_mod(obj.__module__):
      # If this is an indirect module, but the type is part of a submodule which we want to transform,
      # explicitly check the import.
      mod_ = importlib.import_module(ctx.wrapped_mod_prefix + obj.__module__)
      if isinstance(mod_, WrappedSourceModule):
        obj = getattr(mod_, obj.__qualname__)
      else:
        obj = make_wrapped_class(cls=obj, name=name, ctx=ctx)
  elif not isinstance(obj, type) and type(type(obj)) == type:  # object instance
    if type(obj) in ctx.explicit_wrapped_types:
      obj = make_wrapped_object(obj, name=name, ctx=ctx)
    elif not getattr(type(obj), "__module__", None) or not ctx.should_wrap_mod(type(obj).__module__):
      pass  # Don't wrap this object.
    elif isinstance(obj, ctx.keep_as_is_types):  # blacklist
      pass  # keep as-is
    else:
      obj = make_wrapped_object(obj, name=name, ctx=ctx)

  return obj


def unwrap(obj):
  """
  :param obj: potentially wrapped object. could be nested.
  :return: unwrapped object.
    Note that this is not exactly the inverse of :func:`wrap`.
    Namely, not all wrapped objects are unwrapped,
    as this is also not possible in all cases.
  """
  obj = _nested_transform(obj, unwrap)
  if isinstance(obj, WrappedObject):
    # noinspection PyProtectedMember
    return obj._wrapped__orig_obj
  return obj  # leave as-is


def _nested_transform(obj, transform):
  if type(obj) == tuple:
    return tuple([transform(item) for item in obj])
  assert not isinstance(obj, tuple)  # namedtuple or so not implemented yet...
  if type(obj) == list:
    return [transform(item) for item in obj]
  assert not isinstance(obj, list)  # custom subclasses of list not implemented yet...
  if type(obj) == dict:
    # Assume keys are not wrapped.
    return {key: transform(value) for (key, value) in obj.items()}
  if type(obj) == dict:
    # Assume keys are not wrapped.
    return {key: transform(value) for (key, value) in obj.items()}
  if type(obj) == OrderedDict:
    return OrderedDict([(key, transform(value)) for (key, value) in obj.items()])
  if type(obj) == Counter:
    return obj  # as-is
  assert not isinstance(obj, dict)
  return obj


# https://youtrack.jetbrains.com/issue/PY-45382
# noinspection PyTypeChecker
_FuncTypes = (types.FunctionType, types.BuiltinFunctionType)  # type: Tuple[type]
