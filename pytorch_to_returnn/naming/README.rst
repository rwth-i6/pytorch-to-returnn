This provides logic to uniquely give names
to PyTorch modules, and their calls.
This can be used both for original ``torch.nn.Module``'s
(via ``wrapped_import_torch_traced``,
see `import wrappers <../import_wrapper>`__)
or ``pytorch_to_returnn.torch.nn.Module``'s
(e.g. via ``wrapped_import_torch_returnn``,
see `import wrappers <../import_wrapper>`__,
or explicitly used).

The main class to keep track of the names is ``Naming``.
See `naming.py <naming.py>`__.
For custom code, you need to setup a new instance::

  with Naming.make_instance() as naming:
    ...

Every module instance will get a ``ModuleEntry`` instance
(see `module.py <module.py>`__).

Every module call will get a ``CallEntry`` instance
(see `call.py <call.py>`__).

Every tensor instance might get a ``TensorEntry`` instance
(see `tensor.py <tensor.py>`__).

The names will be in ``Naming.root_namespace``,
which is a ``RegisteredName`` instance
(see `namespace.py <namespace.py>`__)
which hierarchically contains the names.

In case we create a RETURNN network and RETURNN layers,
this is stored in a ``ReturnnCtx`` instance
(see `returnn_ctx.py <returnn_ctx.py>`__)
attached to ``RegisteredName``.
