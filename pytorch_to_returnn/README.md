See the [parent readme](..) for an overview.

Sub modules / packages:

* [`import_wrapper`](import_wrapper):
  Python `import` wrapping logic,
  to convert external PyTorch code on-the-fly.

* [`torch`](torch):
  Our drop-in replacement for the PyTorch `torch` package.

* [`naming`](naming.py): (*To be renamed.*)
  Naming logic, and keeping track of context.

* [`verify`](verify.py) (*To be renamed...*)
  PyTorch -> RETURNN conversion (model def and params),
  and verification, whether we get the same outputs.

* [`pprint`](pprint.py):
  Alternative to the official `pprint`.
  Nicer formatting / indentation.
