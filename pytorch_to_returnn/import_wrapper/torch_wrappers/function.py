from .module import WrappedModuleBase


class WrappedTorchFunction(WrappedModuleBase):
  def __init__(self, func, func_name):
    super(WrappedTorchFunction, self).__init__()
    self.func = func
    self.func_name = func_name

  def forward(self, *args, **kwargs):
    return self.func(*args, **kwargs)

  @classmethod
  def has_torch_forward(cls) -> bool:
    # has to be False, such that :func:`_flatten_namespace_for_mod` returns False in `make_module_call` and a
    # corresponding entry in the namespace's `childs_by_name` is created which we need later on to find this module.
    return False
