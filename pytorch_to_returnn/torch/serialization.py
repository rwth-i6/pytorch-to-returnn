

def load(*args, **kwargs):
  # This is just unpickling. It's okay if we get some torch.Tensor or torch.nn.Parameter in the dict.
  # Our code in Module.load_state_dict takes care of that.
  import torch
  return torch.load(*args, **kwargs)


def default_restore_location():
  """Dummy function to avoid import errors"""
  pass
