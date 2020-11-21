

import tensorflow as tf
import torch
import numpy
import types
from pprint import pprint
from typing import Callable, Optional
from returnn.tf.util.data import Data
from .wrapped_import import wrapped_import, wrapped_import_demo
from .import_wrapper.torch_wrappers.module import WrappedModuleBase
from .import_wrapper.torch_wrappers.tensor import WrappedTorchTensor
from .naming import Naming


_InputsType = numpy.ndarray


def verify_torch(
      model_func: Callable[[Optional[Callable[[str], types.ModuleType]], torch.Tensor], torch.Tensor],
      inputs: _InputsType):
  """
  :param model_func: gets one argument wrapped_import(str) -> module, or None. If None, should import as is.
  :param inputs:

  Example code for model func::

    def model_func(wrapped_import, inputs):

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
  torch.manual_seed(42)
  with torch.no_grad():
    out_ref = model_func(None, torch.from_numpy(inputs))
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
  torch.manual_seed(42)
  with torch.no_grad():
    with Naming.make_instance(wrap_to_returnn_enabled=False, keep_orig_module_io_tensors=True) as naming:
      wrapped_torch = wrapped_import("torch")
      out_wrapped = model_func(wrapped_import, wrapped_torch.from_numpy(inputs))
      assert isinstance(out_wrapped, WrappedTorchTensor)
      out_wrapped_np = out_wrapped.cpu().numpy()
      print(">>>> Module naming hierarchy")
      naming.root_namespace.dump()
      print(">>>> Root module calls:")
      pprint(dict(naming.get_root_module_calls()))
      torch_mods_with_params = naming.get_modules_with_params_by_abs_name()
      print(">>>> Modules with params:")
      pprint(dict(torch_mods_with_params))
  assert out_ref_np.shape == out_wrapped_np.shape
  numpy.testing.assert_allclose(out_ref_np, out_wrapped_np)
  print(">>>> Looks good!")
  print()

  print(">>> Running with wrapped Torch import, wrapping replacement for PyTorch...")
  torch.manual_seed(42)
  from . import torch as torch_returnn
  with tf.compat.v1.Session() as session:
    with Naming.make_instance(wrap_to_returnn_enabled=True, keep_orig_module_io_tensors=True) as naming:
      assert isinstance(naming, Naming)
      in_returnn = torch_returnn.from_numpy(inputs)
      assert isinstance(in_returnn, torch_returnn.Tensor)
      n_batch, n_feature, n_time = in_returnn.shape  # currently assumed...
      x = naming.register_input(in_returnn, Data("data", shape=(n_feature, None), feature_dim_axis=1, time_dim_axis=2))
      out_returnn = model_func(wrapped_import_demo, in_returnn)
      assert isinstance(out_returnn, torch_returnn.Tensor)
      y = naming.register_output(out_returnn)
      print("RETURNN output:", y)
      print(">>>> Module naming hierarchy")
      naming.root_namespace.dump()
      print(">>>> Root module calls:")
      pprint(dict(naming.get_root_module_calls()))
      torch_mods_with_params = naming.get_modules_with_params_by_abs_name()
      print(">>>> Modules with params:")
      pprint(dict(torch_mods_with_params))

    session.run(tf.compat.v1.global_variables_initializer())
    feed_dict = {
      x.placeholder: inputs,
      x.get_sequence_lengths(): [n_time] * n_batch}
    y_, y_size = session.run((y.placeholder, y.get_sequence_lengths()), feed_dict=feed_dict)
    print("Output shape:", y_.shape)
    print("Output seq lens:", y_size)
    numpy.testing.assert_allclose(out_ref_np, y_)

  # TODO now build RETURNN model again
  # TODO now forward through RETURNN model
  # TODO check output
  print(">>>> Looks good!")
  print()
