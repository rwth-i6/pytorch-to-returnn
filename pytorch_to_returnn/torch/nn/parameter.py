
from ..tensor import Tensor
from ...naming import Naming


class Parameter(Tensor):
  """
  Distinguish from Tensor.
  """
  def __init__(self, *args, **kwargs):
    super(Parameter, self).__init__(*args, **kwargs)
    naming = Naming.get_instance()
    with naming.push_func_call(func=Parameter, inputs=[]) as ctx:
      naming.register_tensor(self).is_param = True
      ctx.set_outputs([self])
