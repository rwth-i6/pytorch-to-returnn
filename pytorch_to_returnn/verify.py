

import torch
import numpy
import types
from typing import Callable, Optional
from .wrapped_import import wrapped_import


def verify(model_func: Callable[[Optional[Callable[[str], types.ModuleType]]], torch.Tensor]):
  """
  :param model_func: gets one argument wrapped_import(str) -> module.
    model_func will get called multiple times, with different wrapped_import functions.
  """
  # The reference, using the original import.
  out_ref = model_func(None)
  assert isinstance(out_ref, torch.Tensor)
  out_ref_np = out_ref.cpu().numpy()

  # Now with wrapped import. That will also use the original PyTorch code, but wrapped with our custom logic.
  # This should not change anything, and still would use the PyTorch logic,
  # except that the wrapped classes can collect additional information.
  # However, we still will check that we got the same output,
  # just to check that there is no subtle bug due to the wrapping logic.
  # TODO collect information about model?
  out_wrapped = model_func(wrapped_import)
  assert isinstance(out_wrapped, torch.Tensor)  # TODO expect WrappedTensor ...
  out_wrapped_np = out_wrapped.cpu().numpy()

  assert out_ref_np.shape == out_wrapped_np.shape
  numpy.testing.assert_allclose(out_ref_np, out_wrapped_np)

  # TODO: now wrap to wrapped_torch ...
