
import torch
from typing import Optional
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
    with Naming.get_instance().push_module_call(module=self, inputs_args=args, inputs_kwargs=kwargs) as call_entry:
      res = super(WrappedModuleBase, self).__call__(*args, **kwargs)
      call_entry.set_outputs(res)
    return res

  def __setattr__(self, key, value):
    super(WrappedModuleBase, self).__setattr__(key, value)
    if isinstance(value, torch.nn.Module):
      Naming.get_instance().register_module_child_attr(self, key, value)

  def add_module(self, name: str, module: Optional[torch.nn.Module]) -> None:
    super(WrappedModuleBase, self).add_module(name=name, module=module)
    if module:
      Naming.get_instance().register_module_child_attr(self, name, module)

  def get_returnn_name(self) -> str:
    return self.__class__.__name__

  @classmethod
  def has_torch_forward(cls) -> bool:
    in_prefix = False
    from pytorch_to_returnn.import_wrapper.import_ import WrappedModPrefixes
    for prefix in WrappedModPrefixes:
      prefix += "torch.nn."
      if cls.__module__.startswith(prefix):
        in_prefix = True
        break
    if not in_prefix:
      return True
    from pytorch_to_returnn.torch import nn as returnn_torch_nn
    if not hasattr(returnn_torch_nn, cls.__qualname__):
      raise NotImplementedError(f"pytorch_to_returnn.torch.nn.{cls.__qualname__} not yet implemented")
    returnn_cls = getattr(returnn_torch_nn, cls.__qualname__)
    assert issubclass(returnn_cls, returnn_torch_nn.Module)
    return returnn_cls.has_torch_forward()
