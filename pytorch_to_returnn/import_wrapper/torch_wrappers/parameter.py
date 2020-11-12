
import torch
from .tensor import WrappedTorchTensor


class WrappedTorchParameter(WrappedTorchTensor, torch.nn.Parameter):  # TODO
  pass
