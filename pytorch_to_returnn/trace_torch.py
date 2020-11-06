
"""
In user code::

    import torch

    ...

We want to trace all calls to PyTorch functions
(for debugging).

"""

import torch
import sys
import types


_wrap_cache = {}
_printed_getattr = set()


def _print_getattr(full_name: str):
  if full_name in _printed_getattr:
    return
  print("*** getattr %s" % full_name)
  _printed_getattr.add(full_name)



class WrappedModule(types.ModuleType):
  def __init__(self, mod):
    """
    :param types.ModuleType mod:
    """
    super(WrappedModule, self).__init__(name=mod.__name__, doc=mod.__doc__)
    self._mod = mod

  def __getattr__(self, item):
    if item in {"_mod", "__name__"}:
      # noinspection PyUnresolvedReferences
      return super(WrappedModule, self).__getattr__(item)
    res = getattr(self._mod, item)
    res = wrap(res)
    if not isinstance(res, wrapped_types):
      _print_getattr("%s.%s" % (self.__name__, item))
    return res

  def __setattr__(self, key, value):
    if key == "_mod":
      return super(WrappedModule, self).__setattr__(key, value)
    return setattr(self._mod, key, value)


def wrap_module(mod: types.ModuleType) -> WrappedModule:
  if mod in _wrap_cache:
    wrapped_mod = _wrap_cache[mod]
  else:
    wrapped_mod = WrappedModule(mod)
    _wrap_cache[mod] = wrapped_mod
  return wrapped_mod


class WrappedClass:
  pass


class WrappedFunc:
  pass


wrapped_types = (WrappedModule, WrappedClass, WrappedFunc)


def wrap(obj):
  if isinstance(obj, wrapped_types):
    return obj
  if isinstance(obj, types.ModuleType):
    return wrap_module(obj)
  # TODO
  return obj


def enable():
  for name, mod in sorted(sys.modules.items()):
    assert isinstance(name, str)
    if name != "torch" and not name.startswith("torch."):
      continue
    assert isinstance(mod, types.ModuleType)
    if isinstance(mod, WrappedModule):
      print("%s: already wrapped" % name)
      continue
    sys.modules[name] = wrap_module(mod)
