import numpy
from ..modules.rnn import PackedSequence
from ..modules.shape import FlattenBatch
from ...tensor import Tensor
from ..._C import from_numpy
from ....naming import Naming


def pack_padded_sequence(input: Tensor, lengths, batch_first=False, enforce_sorted=True) -> PackedSequence:
  batch_sizes = []
  for frame in range(lengths[0]):
    batch_sizes.append(sum([x > frame for x in lengths]))
  return pack_padded_sequence_with_batch_sizes(input, from_numpy(numpy.array(batch_sizes)), batch_first=batch_first)


def pack_padded_sequence_with_batch_sizes(input: Tensor, batch_sizes: Tensor, batch_first=False) -> PackedSequence:
  return PackedSequence(FlattenBatch(batch_major=batch_first)(input), batch_sizes)


def pad_packed_sequence(sequence: PackedSequence, batch_first=False, padding_value=0.0, total_length=None):
  assert padding_value == 0.0, "not implemented yet"
  assert total_length is None, "not implemented yet"

  return sequence.get_batched_tensor(), sequence.batch_sizes


def pack_sequence(sequences, enforce_sorted=True):
  if isinstance(sequences, Tensor):
    # Note that this is not the intended way to use :func:`pack_sequence`, `sequences` is actually intended to be a list
    # see https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_sequence.html
    #
    # The function also works with a batch-time-major tensor as input, however, then we cannot have different sequence
    # lengths on the PyTorch side and all sequences are assumed to have the same length on the RETURNN side as well.
    naming = Naming.get_instance()
    tensor_entry = naming.tensors[sequences]
    torch_axis_from_returnn_axis = {j: i for i, j in tensor_entry.returnn_axis_from_torch_axis.items()}
    batch_dim = sequences.shape[torch_axis_from_returnn_axis[tensor_entry.returnn_data.batch_dim_axis]]
    time_dim = sequences.shape[torch_axis_from_returnn_axis[tensor_entry.returnn_data.time_dim_axis]]
    lengths = [time_dim] * batch_dim
    return pack_padded_sequence(sequences, lengths, enforce_sorted=enforce_sorted)
  else:
    raise NotImplementedError


def pad_sequence():
  raise NotImplementedError
