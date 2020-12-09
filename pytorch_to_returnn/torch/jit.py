def export(fn):
  """Dummy function to avoid import errors"""
  pass


def unused(fn):
  """Dummy function to avoid import errors"""
  pass


def is_scripting():
  """
  Always returns False as we never are in script mode
  https://pytorch.org/docs/master/jit_language_reference.html#torch.jit.is_scripting

  :return: False
  :rtupe: bool
  """
  return False
