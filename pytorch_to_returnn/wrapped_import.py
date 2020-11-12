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
import typing
from typing import Union, Any
import sys
import importlib
import importlib.abc
import importlib.machinery
from . import log
from pytorch_to_returnn.import_wrapper.base_wrappers.wrapped_object import WrappedObject
from pytorch_to_returnn.import_wrapper.base_wrappers.wrapped_module import WrappedModule
from .import_wrapper.meta_path.loader import MetaPathLoader
from .import_wrapper.meta_path.finder import MetaPathFinder


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
