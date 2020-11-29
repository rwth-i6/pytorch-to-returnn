
from ..tensor import Tensor


def _calculate_fan_in_and_fan_out(tensor: Tensor):
  dimensions = tensor.dim()
  if dimensions < 2:
    raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

  num_input_fmaps = tensor.size(1)
  num_output_fmaps = tensor.size(0)
  receptive_field_size = 1
  if tensor.dim() > 2:
    receptive_field_size = tensor[0][0].numel()
  fan_in = num_input_fmaps * receptive_field_size
  fan_out = num_output_fmaps * receptive_field_size

  return fan_in, fan_out


def zeros_(tensor):
  pass


def uniform_(tensor, a=0., b=1.):
  pass


def normal_(tensor, mean=0., std=1.):
  pass


def xavier_uniform_(tensor: Tensor, gain=1.):
  pass


def xavier_normal_(tensor: Tensor, gain=1.):
  pass


def kaiming_uniform_(tensor: Tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
  pass
