
import torch
from ... import log


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
