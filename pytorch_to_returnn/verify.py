

import torch
import numpy
import types
from typing import Callable, Optional
from .wrapped_import import wrapped_import, wrapped_import_demo


def verify_torch(model_func: Callable[[Optional[Callable[[str], types.ModuleType]]], torch.Tensor]):
  """
  :param model_func: gets one argument wrapped_import(str) -> module, or None. If None, should import as is.

  Example code for model func::

    def model_func(wrapped_import):

        if typing.TYPE_CHECKING or not wrapped_import:
            import torch
            from parallel_wavegan import models as pwg_models
            from parallel_wavegan import layers as pwg_layers

        else:
            torch = wrapped_import("torch")
            wrapped_import("parallel_wavegan")
            pwg_models = wrapped_import("parallel_wavegan.models")
            pwg_layers = wrapped_import("parallel_wavegan.layers")

  model_func will get called multiple times, with different wrapped_import functions.
  wrapped_import would import some user model code.
  wrapped_import expects that the user model code is still unmodified,
  using the original `import torch` statements.

  It will first evaluate model_func with the original imports, i.e. wrapped_import=None.
  Then it will evaluate using ...
  """
  # The reference, using the original import.
  print(">>> Running with standard reference imports...")
  out_ref = model_func(None)
  assert isinstance(out_ref, torch.Tensor)
  out_ref_np = out_ref.cpu().numpy()
  print()

  # Now with wrapped import. That will also use the original PyTorch code, but wrapped with our custom logic.
  # This should not change anything, and still would use the PyTorch logic,
  # except that the wrapped classes can collect additional information.
  # However, we still will check that we got the same output,
  # just to check that there is no subtle bug due to the wrapping logic.
  # TODO collect information about model?
  print(">>> Running with wrapped imports, wrapping original PyTorch...")
  out_wrapped = model_func(wrapped_import)
  assert isinstance(out_wrapped, torch.Tensor)  # TODO expect WrappedTensor ...
  out_wrapped_np = out_wrapped.cpu().numpy()
  assert out_ref_np.shape == out_wrapped_np.shape
  numpy.testing.assert_allclose(out_ref_np, out_wrapped_np)
  print()

  print(">>> Running with wrapped Torch import, wrapping replacement for PyTorch...")
  model_func(wrapped_import_demo)
