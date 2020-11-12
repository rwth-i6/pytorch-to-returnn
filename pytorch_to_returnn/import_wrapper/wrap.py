
from collections import OrderedDict, Counter
import types
import importlib


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


def wrap(obj, name: str):
  if isinstance(obj, (WrappedObject, WrappedModule)):
    return obj
  if isinstance(obj, _KeepAsIsTypes):
    return obj
  obj = _nested_transform(obj, lambda _x: wrap(_x, name="%s..." % name))

  if isinstance(obj, types.ModuleType):
    if _should_wrap_mod(obj.__name__):
      if obj.__name__ not in sys.modules:
        # If this happens, this is actually a bug in how the module importing is done.
        # This is not on our side.
        # This might happen for native Torch modules.
        # This is slightly problematic for our following logic, because import_module will not find it.
        # So we patch this here.
        sys.modules[obj.__name__] = obj
      # If this comes from an WrappedIndirectModule, this will create a new WrappedIndirectModule for the sub mod.
      # See logic in _MetaPathLoader.
      obj = importlib.import_module(_ModPrefix + obj.__name__)
  elif isinstance(obj, (types.FunctionType, types.BuiltinFunctionType)):
    if not obj.__module__ or _should_wrap_mod(obj.__module__):
      obj = make_wrapped_function(func=obj, name=name)
  elif isinstance(obj, type):
    if type(obj) != type:  # explicitly do not check sub-types, e.g. pybind11_type
      pass
    elif obj == torch.Tensor:
      obj = WrappedTorchTensor
    elif obj == torch.nn.Parameter:
      obj = WrappedTorchParameter
    elif obj == torch.nn.Module:
      obj = WrappedModuleBase
    elif obj in _KeepAsIsTypes:
      pass  # keep as-is
    elif obj.__module__ and _should_wrap_mod(obj.__module__):
      # If this is an indirect module, but the type is part of a submodule which we want to transform,
      # explicitly check the import.
      mod_ = importlib.import_module(_ModPrefix + obj.__module__)
      if isinstance(mod_, WrappedSourceModule):
        obj = getattr(mod_, obj.__qualname__)
      else:
        obj = make_wrapped_class(cls=obj, name=name)
  elif not isinstance(obj, type) and type(type(obj)) == type:  # object instance
    if not getattr(obj.__class__, "__module__", None) or not _should_wrap_mod(obj.__class__.__module__):
      pass  # Don't wrap this object.
    elif isinstance(obj, _KeepAsIsTypes):  # blacklist
      pass  # keep as-is
    else:
      obj = make_wrapped_object(obj, name=name)

  return obj


def unwrap(obj):
  obj = _nested_transform(obj, unwrap)
  if isinstance(obj, WrappedObject):
    #if obj._wrapped__orig_obj is obj:
    #  assert False  # TODO
    # noinspection PyProtectedMember
    return obj._wrapped__orig_obj
  return obj  # leave as-is
