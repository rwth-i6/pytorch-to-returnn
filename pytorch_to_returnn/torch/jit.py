def export(fn):
  """Dummy function to avoid import errors"""
  pass


def unused(fn):
  """Dummy function to avoid import errors"""
  pass


def is_scripting() -> bool:
  return False  # stub


def _unwrap_optional(x):
    assert x is not None, "Unwrapping null optional"
    return x

