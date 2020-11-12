
import torch
from ... import log


class WrappedTorchTensor(torch.Tensor):  # TODO
  def __getattribute__(self, item):
    if log.Verbosity >= 6:
      log.unique_print("**** torch tensor __getattribute__ %r" % item)
    return super(WrappedTorchTensor, self).__getattribute__(item)
