
import torch
from ... import log
from ...naming import Naming


class WrappedModuleBase(torch.nn.Module):
  _wrapped_class_cache = {}  # cls -> WrappedClass

  # Need to overwrite to wrap __init__ to correctly set context.
  def __new__(cls, *args, **kwargs):
    if cls not in cls._wrapped_class_cache:
      class WrappedClass(cls):
        def __init__(self, *args, **kwargs):
          self.__class__ = cls  # we don't need this wrapper class anymore
          if log.Verbosity >= 4:
            log.unique_print(
              "*** torch module create %s.%s(...)" % (self.__class__.__module__, self.__class__.__qualname__))
          with Naming.get_instance().push_module_creation(self):
            cls.__init__(self, *args, **kwargs)
      WrappedClass.__name__ = cls.__name__
      WrappedClass.__qualname__ = cls.__qualname__
      WrappedClass.__module__ = cls.__module__
      wrapped_cls = WrappedClass
      cls._wrapped_class_cache[cls] = wrapped_cls
    else:
      wrapped_cls = cls._wrapped_class_cache[cls]
    return super(WrappedModuleBase, cls).__new__(wrapped_cls)

  def __call__(self, *args, **kwargs):
    if log.Verbosity >= 3:
      log.unique_print(
        "*** torch module call %s.%s(...)(...)" % (self.__class__.__module__, self.__class__.__qualname__))
    with Naming.get_instance().push_module_call(module=self, inputs=args):
      return super(WrappedModuleBase, self).__call__(*args, **kwargs)
