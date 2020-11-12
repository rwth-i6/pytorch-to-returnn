"""
Utility to import a module with automatic Torch import wrapping, which replaces all::

  import torch

To::

  from pytorch_to_returnn import torch

In your user code, you would replace::

  import custom_torch_code

By::

  custom_torch_code = pytorch_to_returnn.wrapped_import.wrapped_import("custom_torch_code")

Both the wrapped and original module can be imported at the same time.
The wrapped module will internally get the full mod name ``pytorch_to_returnn._wrapped_mods.custom_torch_code``.
See :class:`_AstImportTransformer`.

"""

import torch
import ast
import types
import typing
from typing import Optional, Union, Any
import sys
from collections import OrderedDict, Counter
import importlib
import importlib.abc
import importlib.machinery
from . import log
from .import_wrapper.ast_transformer import AstImportTransformer


_ExplicitIndirectModList = {
  # Leave "torch" in to not do any magic on the original Torch modules.
  # Note that the main torch module cannot be transformed
  # because of some side-effects triggered by native Torch code,
  # which we cannot catch automatically.
  # More specifically, in torch/__init__.py, there is the call _C._initExtension,
  # which (un-intuitively) does all sorts of initialization,
  # including torch::utils::initializeDtypes,
  # which itself has the side-effect of adding dtypes (e.g. `float32`)
  # to the `torch` module (-> `torch.float32`).
  # And the name `"torch"` is hardcoded there,
  # i.e. it will not used our wrapped module,
  # thus we will never have `float32` registered,
  # and that leads to exceptions.
  # This is likely not the only problem.
  "torch",
}
if "torch" in _ExplicitIndirectModList:
  _ExplicitIndirectModList.update({
    # If we would transform `torch` directly, the following should be explicitly excluded.
    "torch.tensor", "torch.distributed",
    "torch._C",
    # Avoid to have the `Module` class twice.
    "torch.nn.modules.module",
  })

_ExplicitDirectModList = {
  # Note: When we don't wrap all of `torch` (i.e. `"torch"` is in _ExplicitIndirectModList),
  # this will partially be WrappedIndirectModule and WrappedSourceModule.
  # This is not a problem in principle, but there can be subtle problems.
  # Also the import order can be important.
  # E.g. "torch.nn.modules" should be imported before "torch.nn.functional".
  # This might need extra explicit handling, depending what we have here in this list.
  "torch.nn.modules",
  # "torch.nn.functional",  # -- not needed, also causes other problems
}

_KeepAsIsTypes = (
  torch.device,
  torch.Size,
  torch.dtype,
)


def _should_wrap_mod(mod_name: str) -> bool:
  if mod_name == "torch":
    return True
  if mod_name.startswith("torch."):
    return True
  return False


def make_wrapped_object(obj, name: str):
  # First check if we should also wrap the type.
  if _should_wrap_mod(getattr(obj.__class__, "__module__", "")):
    assert getattr(sys.modules[obj.__class__.__module__], obj.__class__.__qualname__) is obj.__class__
    wrapped_mod = importlib.import_module(_ModPrefix + obj.__class__.__module__)
    cls = getattr(wrapped_mod, obj.__class__.__qualname__)
    if cls == WrappedTorchTensor:
      assert isinstance(obj, torch.Tensor)
      obj = obj.as_subclass(cls)
      return obj
    # print("*** wrap obj of type %r with wrapped type %r" % (type(obj), cls))
    assert issubclass(cls, _WrappedClassBase)
    # obj.__class__ = cls  # TODO HACK
    # WrappedObject.__init__(obj, orig_obj=obj, name=name)  # TODO HACK
    # assert obj._wrapped__orig_obj is obj  # ugh...
    obj = cls(orig_obj=obj, name=name)
    return obj
  return WrappedObject(obj, name=name)


class WrappedTorchTensor(torch.Tensor):  # TODO
  def __getattribute__(self, item):
    log.unique_print("**** torch tensor __getattribute__ %r" % item)
    return super(WrappedTorchTensor, self).__getattribute__(item)


class WrappedTorchParameter(WrappedTorchTensor, torch.nn.Parameter):  # TODO
  pass


class WrappedModuleBase(torch.nn.Module):
  def __init__(self):
    super(WrappedModuleBase, self).__init__()
    if log.Verbosity >= 4:
      log.unique_print("*** torch module create %s.%s(...)" % (self.__class__.__module__, self.__class__.__qualname__))

  def __call__(self, *args, **kwargs):
    if log.Verbosity >= 3:
      log.unique_print("*** torch module call %s.%s(...)(...)" % (self.__class__.__module__, self.__class__.__qualname__))
    return super(WrappedModuleBase, self).__call__(*args, **kwargs)

  def __setattr__(self, key, value):
    # value = _unwrap(value)
    return super(WrappedModuleBase, self).__setattr__(key, value)


_TorchModDirectAttribs = {
  "_forward_pre_hooks", "_forward_hooks", "_backward_hooks",
  "_load_state_dict_pre_hooks",
  "_modules", "_parameters", "_buffers",
}


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


def _wrap(obj, name: str):
  if isinstance(obj, (WrappedObject, WrappedModule)):
    return obj
  if isinstance(obj, _KeepAsIsTypes):
    return obj
  obj = _nested_transform(obj, lambda _x: _wrap(_x, name="%s..." % name))

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


def _unwrap(obj):
  obj = _nested_transform(obj, _unwrap)
  if isinstance(obj, WrappedObject):
    #if obj._wrapped__orig_obj is obj:
    #  assert False  # TODO
    # noinspection PyProtectedMember
    return obj._wrapped__orig_obj
  return obj  # leave as-is


class _WrappedClassBase:
  pass


def make_wrapped_class(cls: type, name: str):
  is_torch_module = issubclass(cls, torch.nn.Module)

  # TODO use cls as base?
  class WrappedClass(WrappedObject, _WrappedClassBase):
    def __init__(self, *args, **kwargs):
      if log.Verbosity >= 4:
        log.unique_print("*** WrappedClass %s(...)" % (name,))
      if isinstance(kwargs.get("orig_obj"), cls):
        assert not args
        WrappedObject.__init__(self, **kwargs)
      else:
        obj = cls(*_unwrap(args), **_unwrap(kwargs))
        WrappedObject.__init__(self, orig_obj=obj, name="%s(...)" % name)

    def __getattribute__(self, item):
      #if item == "__dict__":
        # Special case. Directly get it from wrapped object.
      #  return self._wrapped__orig_obj.__dict__
      if item in {"_wrapped__orig_obj", "_wrapped__name", "__class__"}:  # extra checks, and fast path
        return object.__getattribute__(self, item)
      if is_torch_module and False:
        # Some fast path, and avoid logging.
        if item in _TorchModDirectAttribs:
          return getattr(self._wrapped__orig_obj, item)
      if self._wrapped__orig_obj is self:  # special case
        res = object.__getattribute__(self, item)
        res = _wrap(res, name="%s.%s" % (self._wrapped__name, item))
        return res
      return object.__getattribute__(self, item)

    def __setattr__(self, key, value):
      if key in {"_wrapped__orig_obj", "_wrapped__name"}:
        return object.__setattr__(self, key, value)
      if self is self._wrapped__orig_obj:  # special case
        return object.__setattr__(self, key, value)
      return setattr(self._wrapped__orig_obj, key, value)

    if is_torch_module:
      def __call__(self, *args, **kwargs):
        if log.Verbosity >= 3:
          log.unique_print("*** module call %s(...)(...)" % (name,))
        return self._wrapped__orig_obj(*args, **kwargs)

  WrappedClass.__name__ = cls.__name__
  WrappedClass.__qualname__ = cls.__qualname__
  if cls.__module__:
    if _should_wrap_mod(cls.__module__):
      WrappedClass.__module__ = _ModPrefix + cls.__module__
    else:
      WrappedClass.__module__ = cls.__module__
  try:
    WrappedClass.__bases__ += (cls,)
  except TypeError:  # e.g. object layout differs or so
    pass  # just ignore
  return WrappedClass


class WrappedMethod(WrappedObject):  # TODO
  pass


def make_wrapped_function(func, name: str):
  def _call(*args, **kwargs):
    if log.Verbosity >= 3:
      log.unique_print("*** func call %s(...)" % name)
    res = func(*_unwrap(args), **_unwrap(kwargs))
    res = _wrap(res, name="%s(...)" % name)
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


# https://docs.python.org/3/library/importlib.html
# https://www.python.org/dev/peps/pep-0302/
# https://dev.to/dangerontheranger/dependency-injection-with-import-hooks-in-python-3-5hap


_ModPrefix = "%s._wrapped_mods." % __package__


_loader = MetaPathLoader(mod_prefix=_ModPrefix)
_finder = MetaPathFinder(loader=_loader)


def wrapped_import(mod_name: str) -> Union[WrappedModule, Any]:  # rtype Any to make type checkers happy
  """
  :param str mod_name: full mod name, e.g. "custom_torch_code"
  :return: wrapped module
  """
  if _finder not in sys.meta_path:
    sys.meta_path.append(_finder)
  mod = importlib.import_module(_ModPrefix + mod_name)
  assert isinstance(mod, WrappedModule)
  return mod


if typing.TYPE_CHECKING:
  # This also currently does not work...
  # https://youtrack.jetbrains.com/issue/PY-45376
  wrapped_import = importlib.import_module
