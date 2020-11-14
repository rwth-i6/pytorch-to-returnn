
from ..tensor import Tensor
from ...naming import Naming


class Parameter(Tensor):
  """
  Distinguish from Tensor.
  """
  def __init__(self, *args, **kwargs):
    super(Parameter, self).__init__(*args, **kwargs)
    Naming.get_instance().tensors[self].is_param = True
