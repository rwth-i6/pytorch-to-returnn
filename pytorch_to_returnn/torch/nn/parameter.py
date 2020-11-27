
from ..tensor import Tensor
from ...naming import Naming


class Parameter(Tensor):
  """
  Distinguish from Tensor.
  """
  def __init__(self, *args, **kwargs):
    super(Parameter, self).__init__(*args, **kwargs)
    from returnn.tf.util.data import Data
    naming = Naming.get_instance()
    tensor_entry = naming.register_tensor(self)
    tensor_entry.is_param = True
    tensor_entry.returnn_data = Data(
      name="_unnamed_param", shape=self.shape, dtype=self.dtype.name,
      batch_dim_axis=None, time_dim_axis=None)
    tensor_entry.returnn_axis_from_torch_axis = {i: i for i in range(len(self.shape))}
