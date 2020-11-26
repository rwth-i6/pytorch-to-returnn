Generic utility to wrap import logic from some Python code (Python module).
The wrapping happens on the AST level.
(See `class AstImportTransformer <ast_transformer.py>`__ for the technical implementation.)
It will translate ``import <x>`` and ``from <x> import <y>`` statements.

You would use ``wrapped_import(<your_python_mod>)``.
This uses the Python ``meta_path`` logic.
(See `the meta_path sub package <meta_path>`__ for the technical implementation.)

Any such wrapped modules will be instances of ``WrappedSourceModule``.

This is used to replace ``torch`` imports.
E.g. it replaces all::

  import torch

To::

  from pytorch_to_returnn import torch

In your user code, you would replace::

  import custom_torch_code

By::

  custom_torch_code = pytorch_to_returnn.import_wrapper.wrapped_import_torch_returnn("custom_torch_code")

Both the wrapped and original module can be imported at the same time.
The wrapped module will internally get the full mod name
``pytorch_to_returnn.import_wrapper._torch_traced.custom_torch_code``.

Additionally, we also provide utilities to install dynamic proxies
for existing (already loaded) Python modules,
and then also to wrap certain classes and objects on-the-fly.
Such wrapped modules will be instances of ``WrappedIndirectModule``.

This is used to trace calls in the original ``torch`` module,
and to keep track of ``torch.nn.Module`` creations and calls,
and their inputs/outputs.

See the `WrapCtx class <context.py>`__ for details
of what type of wrapping / proxies and dynamic code translation
we support.

This sub package is kept general and can be used for any
such wrapping / proxies and dynamic code translation,
on any external code, on any modules.

We provide two predefined wrapped import functions:

* ``wrapped_import_torch_traced``:
  It will use the original PyTorch ``torch`` module,
  but install dynamic proxy wrappers,
  and trace ``torch.nn.Module`` creation and calls.
  It expects a `Naming context <../naming>`__.
  This needs PyTorch >=1.7 to work properly.

* ``wrapped_import_torch_returnn``:
  It will wrap ``torch`` imports to our ``pytorch_to_returnn.torch``
  (see `the documentation <../torch>`__).
  It expects a `Naming context <../naming>`__.
