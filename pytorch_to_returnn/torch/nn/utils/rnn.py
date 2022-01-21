
from typing import Union, List
from ..functional import full
from ..modules.rnn import PackedSequence
from ..modules.shape import FlattenBatch
from ...tensor import Tensor
from .... import torch
from ....naming import Naming


def _batch_sizes_from_lengths(lengths: Union[Tensor, List[Union[Tensor, int]]]) -> Tensor:
  if not isinstance(lengths, Tensor):
    lengths = torch.convert_to_tensor(lengths)
  indices = torch.arange(torch.max(lengths))  # [T]
  indices_bc = indices.unsqueeze(0)  # [1,T]
  l_bc = lengths.unsqueeze(1)  # [B,1]
  mask = indices_bc < l_bc  # [B,T]
  batch_sizes = torch.sum(mask.int(), dim=0)  # [T]
  return batch_sizes


def pack_padded_sequence(input: Tensor, lengths: Union[Tensor, List[Union[Tensor, int]]],
                         batch_first=False, enforce_sorted=True) -> PackedSequence:
  batch_sizes = _batch_sizes_from_lengths(lengths)
  return pack_padded_sequence_with_batch_sizes(input, batch_sizes, batch_first=batch_first)


def pack_padded_sequence_with_batch_sizes(input: Tensor, batch_sizes: Tensor, batch_first=False) -> PackedSequence:
  # `batch_first` refers to whether the input is a batch major tensor. In the `PackedSequence`, it is always flattened
  # as time major, so we just ignore `batch_first` because RETURNN does not need this information about the input for
  # packing.
  assert batch_first == input.shape[0].is_batch_dim
  return PackedSequence(FlattenBatch(batch_major=False)(input), batch_sizes)


def pad_packed_sequence(sequence: PackedSequence, batch_first=False, padding_value=0.0, total_length=None):
  assert padding_value == 0.0, "not implemented yet"
  assert total_length is None, "not implemented yet"

  return sequence.get_padded_tensor(batch_first=batch_first), sequence.batch_sizes


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
    lengths = full((batch_dim,), time_dim)
    # batch_first is always True because we assume batch-time-major tensor, see above
    return pack_padded_sequence(sequences, lengths, enforce_sorted=enforce_sorted, batch_first=True)
  else:
    raise NotImplementedError


def pad_sequence():
  raise NotImplementedError
