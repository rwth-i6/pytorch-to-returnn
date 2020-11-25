Utility to import a module with automatic Torch import wrapping, which replaces all::

  import torch

To::

  from pytorch_to_returnn import torch

In your user code, you would replace::

  import custom_torch_code

By::

  custom_torch_code = pytorch_to_returnn.import_wrapper.wrapped_import("custom_torch_code")

Both the wrapped and original module can be imported at the same time.
The wrapped module will internally get the full mod name
``pytorch_to_returnn.import_wrapper._traced_torch.custom_torch_code``.
See :class:`AstImportTransformer`.

