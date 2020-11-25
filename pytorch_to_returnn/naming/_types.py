
"""
Only provides types used for type checking, and auto-completion, etc.

Note that this `naming` submodule can be used for different `torch` variants
(e.g. traced original, or our Torch-RETURNN wrapper).
"""

import typing
from typing import Union


if typing.TYPE_CHECKING:
  import torch
  from pytorch_to_returnn.torch.tensor import Tensor as TorchReturnnTensor
  from pytorch_to_returnn.torch.nn import Module as TorchReturnnModule

  # These are the two supported/common cases currently, but this might be extended.
  Tensor = Union[TorchReturnnTensor, torch.Tensor]
  Module = Union[TorchReturnnModule, torch.nn.Module]


# We don't provide the types if not type-checking, by intention.
# This should never be used in actual code (e.g. isinstance checks or so).
# You should always explicitly use the right type in such cases.
