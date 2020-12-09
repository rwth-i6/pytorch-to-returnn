
import torch
from ... import log


# See the discussion here, which is very relevant,
# which covers exactly the topic how one could potentially wrap `torch.Tensor`.
# https://github.com/pytorch/pytorch/blob/master/docs/source/notes/extending.rst#extending-modtorch-with-a-classtensor-like-type

# A soft proxy type unfortunately does not work,
# because `isinstance(obj, torch.Tensor)` checks would not work correctly,
# and they are used in `torch.nn.Module`.


class WrappedTorchTensor(torch.Tensor):
  def __getattribute__(self, item):
    if log.Verbosity >= 10:
      log.unique_print(f"**** torch tensor __getattribute__ {item!r}")
    return super(WrappedTorchTensor, self).__getattribute__(item)

  def new(self, *args, **kwargs):
    # For some reason, new() via __torch_function__ behaves different?
    # This is a workaround.
    if args and isinstance(args[0], torch.Tensor):
      assert len(args) == 1
      assert args[0].type() == self.type()
      return args[0]
    else:
      return self.new_empty(*args, **kwargs)

  @classmethod
  def __torch_function__(cls, func, types, args=(), kwargs=None):
    if log.Verbosity >= 6 and func not in {torch.Tensor.__repr__, torch.Tensor.__str__}:
      log.unique_print(f"**** torch tensor func {func.__name__}")
    if kwargs is None:
      kwargs = {}
    return super().__torch_function__(func, types, args, kwargs)
