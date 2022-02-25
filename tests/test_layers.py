
import _setup_test_env  # noqa
import sys
import unittest
import typing
import numpy
import tensorflow as tf
from pytorch_to_returnn import torch
from pytorch_to_returnn.converter import verify_torch_and_convert_to_returnn
from pytorch_to_returnn.pprint import pformat


def test_randint():
  n_batch, n_time, n_feat = 3, 5, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    out = torch.randint(low=0, high=3, size=(6, 7))
    out = out + 1
    return out

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
    "shape": (None, n_feat), "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": 2})


def test_randint_dynamic():
  n_batch, n_time, n_feat = 3, 5, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    bsz, tsz, fsz = inputs.shape
    out = torch.randint(low=0, high=tsz, size=(bsz, 3 * tsz))
    out = out + 1
    return out

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
    "shape": (None, n_feat), "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": 2})


def test_contrastive_loss():
  n_batch, n_time, n_feat = 3, 14, 7  # B, T', F
  n_negatives = 10  # N

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    model = torch.nn.Conv1d(in_channels=n_feat, out_channels=n_feat, kernel_size=2, stride=3)
    inputs = model(inputs.transpose(1, 2)).transpose(1, 2).contiguous()

    bsz, tsz, fsz = inputs.shape  # (B,T,F)
    tszs = torch.arange(tsz).unsqueeze(-1).expand(-1, n_negatives).flatten()  # (T*N)
    neg_idxs = torch.randint(low=0, high=tsz - 1, size=(bsz, n_negatives * tsz))  # (B,T*N)
    neg_idxs = neg_idxs + (neg_idxs >= tszs).int()  # (B,T*N)
    neg_idxs = neg_idxs + (torch.arange(bsz).unsqueeze(1) * tsz)  # (B,T*N)
    y = inputs.view(-1, fsz)  # (B,T,F) => (B*T,F)
    negs = y[neg_idxs.view(-1)]  # (B*T*N,F)
    negs = negs.view(bsz, tsz, n_negatives, fsz).permute(2, 0, 1, 3)  # to (N,B,T,F)
    inputs_unsqueeze = inputs.unsqueeze(0)  # (1,B,T,F)
    targets = torch.cat([inputs_unsqueeze, negs], dim=0)  # (N+1,B,T,F)
    logits = torch.cosine_similarity(inputs.float(), targets.float(), dim=-1).type_as(inputs)
    labels = logits.new_zeros(logits.size(1) * logits.size(2), dtype=torch.long)
    logits = logits.transpose(0, 2)
    logits = logits.reshape(-1, logits.size(-1))
    output = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
    return output

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time, n_feat)).astype("float32")
  converter = verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
    "shape": (None, n_feat), "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": 2})

  cfg = converter.get_returnn_config_serialized()
  from returnn_helpers import config_net_dict_via_serialized, dummy_run_net
  config, net_dict = config_net_dict_via_serialized(cfg)
  dummy_run_net(config)


def test_embedding():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    model = torch.nn.Embedding(n_in, n_out)
    return model(inputs)

  rnd = numpy.random.RandomState(42)
  x = rnd.randint(0, n_in, (n_time, n_batch), dtype="int64")
  verify_torch_and_convert_to_returnn(
    model_func,
    inputs=x, inputs_data_kwargs={"shape": (None,), "sparse": True, "dim": n_in, "batch_dim_axis": 1})


def test_linear_multiple_steps():
  n_steps = 3
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    ins = inputs.chunk(n_steps, dim=-1)
    model = torch.nn.Linear(n_in, n_out)
    outs = [model(x) for x in ins]
    out = sum(outs)
    return out

  x = numpy.ones((n_batch, n_time, n_in * n_steps)).astype("float32")
  verify_torch_and_convert_to_returnn(
    model_func, inputs=x, inputs_data_kwargs={
      "shape": (None, n_in * n_steps), "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": 2})


def test_load_params_in_returnn():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
      import torch.nn.functional as F
    else:
      torch = wrapped_import("torch")
      F = wrapped_import("torch.nn.functional")

    class DummyModule(torch.nn.Module):
      def __init__(self):
        super(DummyModule, self).__init__()
        self.model = torch.nn.Linear(n_in, n_out)

      def forward(self, x):
        return F.linear(x, self.model.weight)

    mod = DummyModule()
    return mod(inputs)

  x = numpy.ones((n_batch, n_time, n_in)).astype("float32")
  verify_torch_and_convert_to_returnn(
    model_func, inputs=x, inputs_data_kwargs={
      "shape": (None, n_in), "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": 2})


def test_load_params_in_returnn_with_initializer():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
      import torch.nn.functional as F
    else:
      torch = wrapped_import("torch")
      F = wrapped_import("torch.nn.functional")

    class DummyModule(torch.nn.Module):
      def __init__(self):
        super(DummyModule, self).__init__()
        self.model = torch.nn.Linear(n_in, n_out)
        torch.nn.init.xavier_uniform_(self.model.weight)

      def forward(self, x):
        return F.linear(x, self.model.weight)

    mod = DummyModule()
    return mod(inputs)

  x = numpy.ones((n_batch, n_time, n_in)).astype("float32")
  verify_torch_and_convert_to_returnn(
    model_func, inputs=x, inputs_data_kwargs={
      "shape": (None, n_in), "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": 2})


def test_cat():

  def model_func(wrapped_import, inputs: torch.Tensor):
    if wrapped_import:
      torch = wrapped_import("torch")
    else:
      import torch
    return torch.cat((inputs, inputs), dim=-1)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (3, 3)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_cat_non_feature():
  n_batch, n_time, n_feat = 3, 5, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if wrapped_import:
      torch = wrapped_import("torch")
    else:
      import torch

    x = inputs.expand(2, n_batch, n_feat, n_time)
    y = inputs.expand(3, n_batch, n_feat, n_time)
    return torch.cat([x, y], dim=0)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feat, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_conv():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    # {'class': 'transposed_conv', 'from': 'layer2', 'activation': None, 'with_bias': True,
    #  'n_out': 192, 'filter_size': (10,), 'strides': (5,), 'remove_padding': (3,), 'output_padding': (1,)}
    model = torch.nn.Conv1d(
      in_channels=n_in,
      out_channels=n_out,
      kernel_size=3,
      stride=2)
    return model(inputs)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_conv2d():
  n_in, n_out = 11, 13
  n_batch, n_time1, n_time2 = 3, 17, 19

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    # {'class': 'transposed_conv', 'from': 'layer2', 'activation': None, 'with_bias': True,
    #  'n_out': 192, 'filter_size': (10,), 'strides': (5,), 'remove_padding': (3,), 'output_padding': (1,)}
    model = torch.nn.Conv2d(
      in_channels=n_in,
      out_channels=n_out,
      kernel_size=(3, 5),
      stride=2)
    return model(inputs)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time1, n_time2)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_conv_transposed():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    # {'class': 'transposed_conv', 'from': 'layer2', 'activation': None, 'with_bias': True,
    #  'n_out': 192, 'filter_size': (10,), 'strides': (5,), 'remove_padding': (3,), 'output_padding': (1,)}
    model = torch.nn.ConvTranspose1d(
      in_channels=n_in,
      out_channels=n_out,
      kernel_size=10,
      stride=5,
      padding=3,
      output_padding=1)
    return model(inputs)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_conv_transposed_2d():
  n_in, n_out = 11, 13
  n_batch, n_time1, n_time2 = 3, 17, 19

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    model = torch.nn.ConvTranspose2d(
      in_channels=n_in,
      out_channels=n_out,
      kernel_size=(10, 3),
      stride=5,
      padding=(2, 3),
      output_padding=1)
    return model(inputs)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time1, n_time2)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_conv_transposed_2d_with_unsqueeze():
  n_in, n_out = 16, 16
  n_batch, n_features, n_time = 4, 16, 20

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    inputs = inputs.unsqueeze(-1)  # (B, F, T, 1)
    model = torch.nn.ConvTranspose2d(
      in_channels=n_in,
      out_channels=n_out,
      kernel_size=(1, 12))
    return model(inputs)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_features, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_functional_linear():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
      import torch.nn.functional as F
    else:
      torch = wrapped_import("torch")
      F = wrapped_import("torch.nn.functional")
    rnd = numpy.random.RandomState(42)
    weight = rnd.normal(0., 1., (n_out, n_in)).astype("float32")
    bias = rnd.normal(0., 1., (n_out,)).astype("float32")
    weight = torch.from_numpy(weight)
    bias = torch.from_numpy(bias)
    return F.linear(inputs.transpose(1, 2), weight=weight, bias=bias)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_multiplication_broadcasting():
  n_batch, n_time, n_feature = 3, 7, 11

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    bsz, fsz, tsz = inputs.shape
    out = inputs * inputs.expand(3, bsz, fsz, tsz)
    assert out.shape == (3, bsz, fsz, tsz)
    return out

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feature, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_matmul_broadcasting():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
      import torch.nn.functional as F
    else:
      torch = wrapped_import("torch")
      F = wrapped_import("torch.nn.functional")
    rnd = numpy.random.RandomState(42)
    weight = rnd.normal(0., 1., (n_in, n_out)).astype("float32")
    weight = torch.from_numpy(weight)
    return torch.matmul(inputs.transpose(1, 2), weight)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_matmul_shared_remaining_axes():
  n_1, n_2 = 2, 4
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    return torch.matmul(inputs, inputs.transpose(2, 3))

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_1, n_time, n_2)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
      "shape": (n_1, None, n_2), "batch_dim_axis": 0, "time_dim_axis": 2, "feature_dim_axis": 3})


def test_spatial_axes_with_same_tag():
  n_1, n_2 = 2, 4
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
      import torch.nn.functional as F
    else:
      torch = wrapped_import("torch")
      F = wrapped_import("torch.nn.functional")
    x = torch.matmul(inputs, inputs.transpose(2, 3))
    x = F.softmax(x, dim=-1)
    return x

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_1, n_time, n_2)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
      "shape": (n_1, None, n_2), "batch_dim_axis": 0, "time_dim_axis": 2, "feature_dim_axis": 3})


def test_bmm():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 5

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")

    inputs = inputs.transpose(0, 1)
    x = inputs.new_zeros(inputs.shape[0], inputs.shape[2], n_out) + 1  # (B, F_in, F_out)
    return torch.bmm(inputs, x)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_time, n_batch, n_in)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
      "shape": (None, n_in), "time_dim_axis": 0, "batch_dim_axis": 1, "feature_dim_axis": 2})


def test_packed_sequence_1():
  """
  Regular packing and unpacking from batched, padded tensor
  """
  n_batch, n_time, n_feat = 3, 5, 7
  seq_lens = [5, 4, 3]

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")

    h = torch.nn.utils.rnn.pack_padded_sequence(inputs, seq_lens)
    output, _ = torch.nn.utils.rnn.pad_packed_sequence(h)
    return output + 1

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_time, n_batch, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, returnn_input_seq_lens={0: seq_lens}, inputs_data_kwargs={
      "shape": (None, n_feat), "time_dim_axis": 0, "batch_dim_axis": 1, "feature_dim_axis": 2})


def test_packed_sequence_2():
  """
  Packing and unpacking from batched, padded tensor, where the packing is done with :func:`pack_sequence` which actually
  requires a list of tensors as input.
  """
  n_batch, n_time, n_feat = 3, 5, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")

    if not getattr(torch, "__returnn__", False):
      inputs = list(inputs)
    h = torch.nn.utils.rnn.pack_sequence(inputs)
    output, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
    return output + 1

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
      "shape": (None, n_feat), "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": 2})


def test_packed_sequence_3():
  """
  Pack and return packed .data
  """
  n_batch, n_time, n_feat = 3, 5, 7
  seq_lens = [5, 4, 3]

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")

    h = torch.nn.utils.rnn.pack_padded_sequence(inputs, seq_lens)
    return h.data

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_time, n_batch, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, returnn_input_seq_lens={0: seq_lens}, inputs_data_kwargs={
      "shape": (None, n_feat), "time_dim_axis": 0, "batch_dim_axis": 1, "feature_dim_axis": 2})


def test_packed_sequence_4():
  """
  Initialize :class:`PackedSequence` directly using its init
  """
  n_batch, n_time, n_feat = 3, 5, 7
  seq_lens = [5, 4, 3]

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")

    h = torch.nn.utils.rnn.pack_padded_sequence(inputs, seq_lens)
    h = torch.nn.utils.rnn.PackedSequence(h.data, h.batch_sizes)
    return h.data

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_time, n_batch, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, returnn_input_seq_lens={0: seq_lens}, inputs_data_kwargs={
      "shape": (None, n_feat), "time_dim_axis": 0, "batch_dim_axis": 1, "feature_dim_axis": 2})


def test_packed_sequence_5():
  """
  Pack and return packed .data, like :func:`test_packed_sequence_3` but with batch major input
  """
  n_batch, n_time, n_feat = 3, 5, 7
  seq_lens = [5, 4, 3]

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")

    h = torch.nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=True)
    return h.data

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, returnn_input_seq_lens={0: seq_lens}, inputs_data_kwargs={
      "shape": (None, n_feat), "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": 2})


def test_packed_sequence_6():
  """
  Pack with `batch_first=False` and unpack with `batch_first=True`
  """
  n_batch, n_time, n_feat = 3, 5, 7
  seq_lens = [5, 4, 3]

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")

    h = torch.nn.utils.rnn.pack_padded_sequence(inputs, seq_lens)
    output, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
    return output + 1

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_time, n_batch, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, returnn_input_seq_lens={0: seq_lens}, inputs_data_kwargs={
      "shape": (None, n_feat), "time_dim_axis": 0, "batch_dim_axis": 1, "feature_dim_axis": 2})


def test_lstm_with_packed_sequence_input():
  n_batch, n_time, n_feat = 3, 5, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")

    h = torch.nn.utils.rnn.pack_padded_sequence(inputs, [n_time] * n_batch)
    output, _ = torch.nn.LSTM(n_feat, n_feat)(h)
    return output.data

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_time, n_batch, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
      "shape": (None, n_feat), "time_dim_axis": 0, "batch_dim_axis": 1, "feature_dim_axis": 2})


def test_t():
  n_batch, n_feature, n_time = 3, 5, 17

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
      import torch.nn.functional as F
    else:
      torch = wrapped_import("torch")
      F = wrapped_import("torch.nn.functional")
    rnd = numpy.random.RandomState(42)
    weight = rnd.normal(0., 1., (3, 5)).astype("float32")
    weight = torch.from_numpy(weight)
    weight = weight.t()
    return F.relu(inputs)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feature, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_reshape_ab_c_to_a_bc():
  n_batch, n_time, n_feature_1, n_feature_2 = 3, 5, 8, 6

  def model_func(wrapped_import, inputs: torch.Tensor):
    # test case (..., a*b, c,...) -> (..., a, b*c,...)
    return inputs.view(n_batch, n_time, n_feature_1 // 2, n_feature_2 * 2)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time, n_feature_1, n_feature_2)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
      "shape": (None, n_feature_1, n_feature_2), "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": 2})


def test_reshape_a_bc_to_ab_c():
  n_batch, n_time, n_feature_1, n_feature_2 = 3, 5, 8, 6

  def model_func(wrapped_import, inputs: torch.Tensor):
    # test case (..., a, b*c,...) -> (..., a*b, c,...)
    return inputs.view(n_batch, n_time, n_feature_1 * 2, n_feature_2 // 2)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time, n_feature_1, n_feature_2)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
      "shape": (None, n_feature_1, n_feature_2), "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": 2})


def test_reshape_a_b_F_to_b_aF():
  n_batch, n_time, n_feature_1, n_feature_2 = 3, 5, 8, 6

  def model_func(wrapped_import, inputs: torch.Tensor):
    # test case (..., a, b, F,...) -> (..., b, a*F,...)
    inputs = inputs.transpose(1, 2).contiguous()
    return inputs.view(n_batch, n_time, n_feature_1 * n_feature_2)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feature_1, n_time, n_feature_2)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
      "shape": (n_feature_1, None, n_feature_2), "batch_dim_axis": 0, "time_dim_axis": 2, "feature_dim_axis": 3})


def test_reshape_a_b_1_to_a_b():
  n_batch, n_time, n_feature = 2, 7, 1

  def model_func(wrapped_import, inputs: torch.Tensor):
    # test case (a, b, 1) -> (a, b)
    out = inputs.view(inputs.shape[:2])  # (B, T, F)
    return out

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time, n_feature)).astype("float32")
  verify_torch_and_convert_to_returnn(
    model_func, inputs=x, returnn_dummy_input_shape=x.shape,
    inputs_data_kwargs={
      "shape": (None, n_feature), "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": 2})


def test_pad():
  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch.nn.functional as F
    else:
      F = wrapped_import("torch.nn.functional")

    inputs = F.pad(inputs, (1, 1, 2, 2))
    inputs = F.pad(inputs, (1, 1))
    return inputs

  x = numpy.zeros((1, 1, 4, 4)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_pad_time_btf():
  n_batch, n_time, n_feat = 3, 7, 5

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch.nn.functional as F
    else:
      F = wrapped_import("torch.nn.functional")

    inputs = F.pad(inputs, (0, 0, 2, 2))
    return inputs

  x = numpy.ones((n_batch, n_time, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
    "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": 2, "shape": (None, n_feat)})


def test_constant_pad_1d():
  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")

    mod = torch.nn.ConstantPad1d((4, 1), 0)
    return mod(inputs)

  x = numpy.zeros((3, 5, 7)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_functional_conv():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7
  kernel_size = 3

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
      import torch.nn.functional as F
    else:
      torch = wrapped_import("torch")
      F = wrapped_import("torch.nn.functional")
    rnd = numpy.random.RandomState(42)
    weight = rnd.normal(0., 1., (n_out, n_in, kernel_size)).astype("float32")
    bias = rnd.normal(0., 1., (n_out,)).astype("float32")
    weight = torch.from_numpy(weight)
    bias = torch.from_numpy(bias)
    return F.conv1d(
      inputs,
      weight=weight, bias=bias,
      stride=2)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_functional_conv_no_bias():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7
  kernel_size = 3

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
      import torch.nn.functional as F
    else:
      torch = wrapped_import("torch")
      F = wrapped_import("torch.nn.functional")
    rnd = numpy.random.RandomState(42)
    weight = rnd.normal(0., 1., (n_out, n_in, kernel_size)).astype("float32")
    weight = torch.from_numpy(weight)
    return F.conv1d(inputs, weight=weight, stride=2)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_functional_conv2d():
  n_in, n_out = 11, 13
  n_batch, n_time1, n_time2 = 3, 17, 19
  kernel_size = (3, 5)

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
      import torch.nn.functional as F
    else:
      torch = wrapped_import("torch")
      F = wrapped_import("torch.nn.functional")
    rnd = numpy.random.RandomState(42)
    weight = rnd.normal(0., 1., (n_out, n_in) + kernel_size).astype("float32")
    bias = rnd.normal(0., 1., (n_out,)).astype("float32")
    weight = torch.from_numpy(weight)
    bias = torch.from_numpy(bias)
    return F.conv2d(
      inputs,
      weight=weight, bias=bias,
      stride=2)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time1, n_time2)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_functional_conv_transposed():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7
  kernel_size = 3

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
      import torch.nn.functional as F
    else:
      torch = wrapped_import("torch")
      F = wrapped_import("torch.nn.functional")
    rnd = numpy.random.RandomState(42)
    weight = rnd.normal(0., 1., (n_in, n_out, kernel_size)).astype("float32")
    weight = torch.from_numpy(weight)
    return F.conv_transpose1d(inputs, weight=weight, stride=2)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_batch_norm():
  n_in, n_batch, n_time = 11, 3, 7
  for train in [True, False]:

    def model_func(wrapped_import, inputs: torch.Tensor):
      if typing.TYPE_CHECKING or not wrapped_import:
        import torch
      else:
        torch = wrapped_import("torch")
      model = torch.nn.BatchNorm1d(n_in)
      if not train:
        model.eval()
      out = model(inputs)
      if train:
        model.reset_running_stats()  # for the test, such that we start with initial running mean/var
      return out

    x = numpy.ones((n_batch, n_in, n_time)).astype("float32")
    verify_torch_and_convert_to_returnn(model_func, inputs=x, train=train)

    rnd = numpy.random.RandomState(42)
    x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
    verify_torch_and_convert_to_returnn(model_func, inputs=x, train=train)


def test_batch_norm_running_stats():
  from pytorch_to_returnn.torch.nn import Module as ModuleReturnn
  from pytorch_to_returnn.naming import Naming, ModuleEntry
  from torch.nn import Module as ModuleTorch
  n_in, n_batch, n_time = 11, 3, 7
  mean_torch = None
  mean_returnn = None

  def model_func(wrapped_import, inputs: torch.Tensor):
    nonlocal mean_torch, mean_returnn
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    model = torch.nn.BatchNorm1d(n_in)
    out = model(inputs)
    if isinstance(model, ModuleTorch):
      mean_torch = model.running_mean.detach().cpu().numpy().copy()
    elif isinstance(model, ModuleReturnn):
      naming = Naming.get_instance()
      if naming.import_params_from_torch_namespace:  # only then we have the params
        module_entry = naming.modules[model]
        assert isinstance(module_entry, ModuleEntry)
        assert len(module_entry.calls) == 1
        call = module_entry.calls[0]
        assert call.returnn_layer
        mean_returnn = tf.squeeze(call.returnn_layer.params["batch_norm/v2_mean"]).eval()
    model.reset_running_stats()  # for the test, such that we start with initial running mean/var
    return out

  x = numpy.ones((n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, train=True)
  assert mean_returnn is not None and mean_torch is not None
  print(mean_torch)
  numpy.testing.assert_allclose(mean_torch[0], 0.1, rtol=0, atol=1e-5)  # default momentum 0.1
  numpy.testing.assert_allclose(mean_returnn, mean_torch, rtol=0, atol=1e-5)

  mean_returnn = mean_torch = None
  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, train=True)
  assert mean_returnn is not None and mean_torch is not None
  print(mean_torch)
  numpy.testing.assert_allclose(mean_returnn, mean_torch, rtol=0, atol=1e-5)


def test_fp32_layer_norm():
  n_in, n_batch, n_time = 11, 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
      import torch.nn.functional as F
    else:
      torch = wrapped_import("torch")
      F = wrapped_import("torch.nn.functional")

    # copy of Fp32LayerNorm from fairseq
    class Fp32LayerNorm(torch.nn.LayerNorm):
      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

      def forward(self, input):
        output = F.layer_norm(
          input.float(),
          self.normalized_shape,
          self.weight.float() if self.weight is not None else None,
          self.bias.float() if self.bias is not None else None,
          self.eps
        )
        return output.type_as(input)

    model = Fp32LayerNorm(n_in, elementwise_affine=True)
    out = inputs.transpose(-2, -1)
    out = model(out)
    out = out.transpose(-2, -1)
    return out

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
    "batch_dim_axis": 0, "time_dim_axis": 2, "feature_dim_axis": 1, "shape": (n_in, None)})


def test_group_norm():
  n_batch, n_time = 4, 20
  for num_groups, num_channels in [(1, 5), (5, 5)]:

    def model_func(wrapped_import, inputs: torch.Tensor):
      if typing.TYPE_CHECKING or not wrapped_import:
        import torch
      else:
        torch = wrapped_import("torch")

      model = torch.nn.GroupNorm(num_groups, num_channels)
      out = model(inputs)
      return out

    print(f"test for num_groups={num_groups}, num_channels={num_channels}")
    rnd = numpy.random.RandomState(42)
    x = rnd.normal(0., 1., (n_batch, num_channels, n_time)).astype("float32")
    verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_fp32_group_norm():
  n_batch, n_time = 3, 17
  for num_groups, num_channels in [(1, 5), (5, 5)]:

    def model_func(wrapped_import, inputs: torch.Tensor):
      if typing.TYPE_CHECKING or not wrapped_import:
        import torch
        import torch.nn.functional as F
      else:
        torch = wrapped_import("torch")
        F = wrapped_import("torch.nn.functional")

      # copy of Fp32GroupNorm from fairseq
      class Fp32GroupNorm(torch.nn.GroupNorm):
        def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)

        def forward(self, input):
          output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps)
          return output.type_as(input)

      model = Fp32GroupNorm(num_groups, num_channels)
      out = model(inputs)
      return out

    print(f"test for num_groups={num_groups}, num_channels={num_channels}")
    rnd = numpy.random.RandomState(42)
    x = rnd.normal(0., 1., (n_batch, num_channels, n_time)).astype("float32")
    verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_fp32_group_norm_subnetwork():
  n_batch, n_time = 3, 17
  for num_groups, num_channels in [(1, 5), (5, 5)]:

    def model_func(wrapped_import, inputs: torch.Tensor):
      if typing.TYPE_CHECKING or not wrapped_import:
        import torch
        import torch.nn.functional as F
      else:
        torch = wrapped_import("torch")
        F = wrapped_import("torch.nn.functional")

      # copy of Fp32GroupNorm from fairseq
      class Fp32GroupNorm(torch.nn.GroupNorm):
        def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)

        def forward(self, input):
          output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps)
          return output.type_as(input)

      model = torch.nn.Sequential(Fp32GroupNorm(num_groups, num_channels))
      out = model(inputs)
      return out

    print(f"test for num_groups={num_groups}, num_channels={num_channels}")
    rnd = numpy.random.RandomState(42)
    x = rnd.normal(0., 1., (n_batch, num_channels, n_time)).astype("float32")
    verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_multi_head_attention_forward():
  n_batch, n_time, n_feature = 3, 17, 8

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
      import torch.nn.functional as F
    else:
      torch = wrapped_import("torch")
      F = wrapped_import("torch.nn.functional")

    class MultiHeadAttention(torch.nn.Module):
      def __init__(self, embed_dim, num_heads,):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dummy_weight = torch.ones(embed_dim, embed_dim)
        self.dummy_in_bias = torch.zeros(3 * embed_dim)
        self.dummy_out_bias = torch.zeros(embed_dim)

        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.k_proj.weight)
        torch.nn.init.xavier_uniform_(self.v_proj.weight)
        torch.nn.init.xavier_uniform_(self.q_proj.weight)

      def forward(self, query, key, value):
        attn_output, attn_output_weights = F.multi_head_attention_forward(
          query, key, value, self.embed_dim, self.num_heads,
          in_proj_weight=self.dummy_weight, in_proj_bias=self.dummy_in_bias,
          bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0.0,
          out_proj_weight=self.dummy_weight, out_proj_bias=self.dummy_out_bias,
          need_weights=False, use_separate_proj_weight=True,
          k_proj_weight=self.k_proj.weight, q_proj_weight=self.q_proj.weight, v_proj_weight=self.v_proj.weight)
        return attn_output


    model = MultiHeadAttention(embed_dim=n_feature, num_heads=2)
    inputs = inputs
    out = model(query=inputs, key=inputs, value=inputs)
    return out

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_time, n_batch, n_feature)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
    "time_dim_axis": 0, "batch_dim_axis": 1, "feature_dim_axis": 2, "shape": (None, n_feature)})


def test_cosine_similarity():
  n_batch, n_time, n_feat = 3, 5, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if wrapped_import:
      torch = wrapped_import("torch")
    else:
      import torch

    x = inputs
    y = inputs + 1
    ref = torch.cosine_similarity(x, y, dim=2)
    return ref

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
    "shape": (None, n_feat), "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": 2})


def test_cross_entropy():
  n_batch, n_feat = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if wrapped_import:
      torch = wrapped_import("torch")
    else:
      import torch

    target = torch.randint(low=0, high=inputs.shape[0], size=(inputs.shape[0],))
    ref = torch.nn.functional.cross_entropy(inputs, target, reduction="sum")
    return ref

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
    "shape": (n_feat,), "batch_dim_axis": 0, "feature_dim_axis": 1, "time_dim_axis": None})


def test_unsqueeze():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    return inputs.unsqueeze(2)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_unsqueeze2():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    return inputs.unsqueeze(3)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_unsqueeze3():
  from pytorch_to_returnn.naming import Naming
  n_batch, n_time = 3, 7
  n_in, n_out = 1, 4

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    out = inputs.unsqueeze(1)
    model = torch.nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=2)
    out = model(out)
    return out

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
      "shape": (None,), "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": None})


def test_transpose_unsqueeze():
  n_batch, n_time, n_feat = 3, 5, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    x = inputs.transpose(1, 2)
    x = x.unsqueeze(0)
    return x

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feat, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_broadcast_with_different_axes_types():
  n_batch, n_time, n_feature = 3, 7, 5

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    weight = torch.ones(n_feature)
    return inputs * weight.view(1, -1, 1)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feature, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_broadcast_after_transpose():
  n_batch, n_time, n_feature = 3, 7, 5

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    weight = torch.ones(n_feature)  # [F]
    x = inputs.transpose(1, 2)  # [B, T, F]
    return x * weight

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feature, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_broadcast_add_bias():
  n_batch, n_feature = 3, 5

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    rnd = numpy.random.RandomState(42)
    bias = rnd.normal(0., 1., (n_feature,)).astype("float32")
    bias = torch.from_numpy(bias)
    return inputs + bias

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feature)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={"time_dim_axis": None})


def test_new_zeros():
  n_batch, n_time, n_in = 3, 5, 11

  def model_func(wrapped_import, inputs: torch.Tensor):
    x = inputs.new_zeros(inputs.shape)
    return inputs + x

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time, n_in)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
      "shape": (None, n_in), "time_dim_axis": 1, "batch_dim_axis": 0, "feature_dim_axis": 2})


def test_movedim():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    x = inputs  # (B,F,T)
    x = torch.nn.Conv1d(n_in, n_out, 3)(x)  # make it (B,T,F) in RETURNN. (B,F,T) in Torch.
    x = torch.movedim(x, 0, 1)  # stay (B,T,F) in RETURNN. (F,B,T) in Torch.
    return x

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_movedim_2():
  n_batch, n_time, n_feature = 3, 5, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    out = torch.movedim(inputs, 0, 2)
    return out

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feature, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_permute():
  n_batch, n_time, n_feature = 3, 5, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    out = inputs.permute(2, 0, 1)
    return out

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feature, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_flatten_batch():
  n_in, n_out = 11, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    x = inputs  # (B,F,T)
    x = torch.movedim(x, 1, 0)  # (F,B,T)
    x = torch.reshape(x, (x.shape[0], -1))  # (F,B*T)
    return x

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_merge_batch_with_modified_time():
  n_in, n_out = 5, 7
  n_batch, n_time = 3, 11

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")

    conv = torch.nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=3, stride=2)
    y = inputs  # (B,F,T)
    y = conv(y)  # (B,F',T')
    y = y.transpose(1, 2).contiguous()  # (B,T',F')
    _, _, fsz = y.shape
    y = y.view(-1, fsz)  # (B*T',F')
    return y

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_reduce_sum():
  n_batch, n_time, n_feature = 3, 5, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    out = inputs.sum(dim=1)
    return out

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feature, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_slice_1d():
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    x = inputs[0]  # (T)
    x = x[:3]
    return x

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_slice_2d():
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    x = inputs[:, 3:]
    return x

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_slice_tensor():
  n_batch, n_time, n_in = 3, 5, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    y = inputs.view(-1, n_in)
    idx = torch.arange(4) + 2
    out = y[idx]
    return out

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time, n_in)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
    "shape": (None, n_in), "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": 2})


def test_index_none():
  n_batch, n_time, n_feat = 3, 5, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    x = inputs[None]
    return x

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feat, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_index_merged_dim():
  n_batch, n_time, n_in = 3, 5, 7
  n_index = 2

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")

    n_b, n_t, n_f = inputs.shape
    y = inputs.view(-1, n_f)  # [B*T, F]
    idx = torch.arange(n_b * n_t)  # [B*T] -> indices in B*T
    idx = torch.cat([idx] * n_index)  # [2*B*T]
    idx = idx.reshape(n_b, n_t * n_index)  # [B, 2*T]
    idx = idx.view(-1)  # [2*B*T]
    x = y[idx]  # [2*B*T, F]
    x = x.view(n_b, n_t, n_index, n_f)  # [B,T,2,F]
    x = x.permute(2, 0, 1, 3)  # [2,B,T,F]
    out = torch.cat([inputs.unsqueeze(0), x])  # [3,B,T,F]
    return out

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time, n_in)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
    "shape": (None, n_in), "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": 2})


def test_expand():
  n_batch, n_feat = 5, 1

  def model_func(wrapped_import, inputs: torch.Tensor):
    x = inputs.expand(3, 2, -1, 7)
    return x

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_arange():
  n_batch, n_feat = 1, 5

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    arange = torch.arange(inputs.shape[1])
    return inputs + torch.reshape(arange, (1, -1))

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_arange_dyn():
  n_batch, n_feat = 3, 5

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    arange = torch.arange(inputs.shape[0])
    return arange

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_arange_dyn_unsqueeze():
  n_batch, n_feat = 3, 5

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    arange = torch.arange(inputs.shape[0])
    arange = arange.unsqueeze(1)
    return arange

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_arange_dyn_unsqueeze_add():
  n_batch, n_feat = 3, 5

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    randint = torch.randint(low=0, high=10, size=(inputs.shape[0], 4))
    arange = torch.arange(inputs.shape[0]).unsqueeze(1)
    return randint + arange

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_arange_from_lengths():
  n_batch, n_time = 3, 5

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    batch_dim, time_dim = inputs.shape
    lengths = torch.full((batch_dim,), time_dim)
    arange = torch.arange(torch.max(lengths))
    assert arange.shape == (n_time,)
    return arange

  rnd = numpy.random.RandomState(42)
  x = rnd.randint(0, 10, (n_batch, n_time), dtype="int64")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
    "shape": (None,), "batch_dim_axis": 0, "time_dim_axis": 1})


def test_broadcasting_with_lengths():
  n_batch, n_time = 3, 5

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    batch_dim, time_dim = inputs.shape
    lengths = torch.full((batch_dim,), time_dim)
    indices = torch.arange(torch.max(lengths))
    indices_bc = indices.unsqueeze(0)  # [1,T]
    l_bc = lengths.unsqueeze(1)  # [B,1]
    mask = indices_bc < l_bc  # [B,T]
    return mask

  rnd = numpy.random.RandomState(42)
  x = rnd.randint(0, 10, (n_batch, n_time), dtype="int64")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
    "shape": (None,), "batch_dim_axis": 0, "time_dim_axis": 1})


def test_full():
  n_batch, n_feat = 3, 5

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    return torch.full((inputs.shape[0],), 42)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_full_out_dim_tag():
  n_batch, n_feat = 3, 5

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    full = torch.full((inputs.shape[0],), 42)
    arange = torch.arange(full.shape[0])
    return arange + full

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feat)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_const_with_batch_and_gather():
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    batch_size = inputs.shape[0]
    x = inputs.new_zeros((2, batch_size, 13))  # const, shape (2,B,13). in RETURNN (B,2,13)
    x = x[0]  # shape (B,13). in RETURNN (B,13).
    x = x + inputs[:, :1]  # (B,13). just do this to have dep on inputs
    return x

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_lstm_2l():
  n_batch, n_time = 3, 7
  n_in, n_out = 11, 13
  n_layers = 2

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    batch_size = inputs.shape[1]
    hidden = inputs.new_zeros((n_layers, batch_size, n_out))  # const, shape (2,B,13). in RETURNN (B,2,13)
    hidden = (hidden, hidden)
    lstm = torch.nn.LSTM(n_in, n_out, n_layers)
    out, hidden_ = lstm(inputs, hidden)
    return out

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_time, n_batch, n_in)).astype("float32")
  verify_torch_and_convert_to_returnn(
    model_func, inputs=x,
    inputs_data_kwargs={"shape": (None, n_in), "batch_dim_axis": 1})


def test_blstm():
  n_batch, n_time = 3, 7
  n_in, n_out = 11, 13
  n_layers = 1

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    batch_size = inputs.shape[1]
    hidden = inputs.new_zeros((2 * n_layers, batch_size, n_out))
    hidden = (hidden, hidden)
    blstm = torch.nn.LSTM(n_in, n_out, n_layers, bidirectional=True)
    out, hidden_ = blstm(inputs, hidden)
    return out

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_time, n_batch, n_in)).astype("float32")
  verify_torch_and_convert_to_returnn(
    model_func, inputs=x,
    inputs_data_kwargs={"shape": (None, n_in), "batch_dim_axis": 1})


def test_multiple_outputs():
  n_batch, n_time = 3, 7
  n_in, n_out = 11, 13
  n_layers = 1

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")

    class InnerModel(torch.nn.Module):
      def __init__(self):
        super(InnerModel, self).__init__()
        self.lstm = torch.nn.LSTM(n_in, n_out, n_layers)

      def forward(self, x, hidden):
        return self.lstm(x, hidden)

    # Wrap InnerModel in another model, to make sure InnerModel is not on the flattened root namespace.
    class OuterModel(torch.nn.Module):
      def __init__(self):
        super(OuterModel, self).__init__()
        self.model = InnerModel()

      def forward(self, input):
        batch_size = input.shape[1]
        hidden = torch.zeros((n_layers, batch_size, n_out))  # const, shape (1,B,13). in RETURNN (B,1,13)
        hidden = (hidden, hidden)
        out, (h, c) = self.model(input, hidden)  # out is (T,B,out), h/c is (1,B,out)
        return out[0] + h[0] + c[0]  # combine all, to set dependencies. (B,out)

    model = OuterModel()
    return model(inputs)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_time, n_batch, n_in)).astype("float32")
  verify_torch_and_convert_to_returnn(
    model_func, inputs=x,
    inputs_data_kwargs={"shape": (None, n_in), "batch_dim_axis": 1})


def test_output_type():
  n_batch, n_time, n_feat = 3, 5, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if wrapped_import:
      torch = wrapped_import("torch")
    else:
      import torch


    class DummyModule(torch.nn.Module):
      def forward(self, y):
        return {"a": 1, "b": y + 1}

    mod = DummyModule()
    return mod(inputs)["b"]

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feat, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_forward_with_kwargs():
  n_batch, n_time, n_feature = 3, 7, 5

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")

    class MyModel(torch.nn.Module):
      def __init__(self, bias=0):
        super(MyModel, self).__init__()
        self.bias = bias

      def forward(self, x, *, add_bias=None):
        assert isinstance(add_bias, bool)
        if add_bias:
          x += self.bias
        return x

    model = MyModel(bias=1)
    out = model(inputs, add_bias=True)
    return out

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feature, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_gated_linear_unit_via_chunk():
  n_in, n_out = 12, 13
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")

    class GLU(torch.nn.Module):
      def __init__(self, dim):
        super().__init__()
        self.dim = dim

      def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

    model = GLU(dim=1)
    return model(inputs)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_depth_wise_conv1d():
  n_in, n_out = 12, 24
  n_batch, n_time = 3, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
      import torch.nn.functional as F
    else:
      torch = wrapped_import("torch")
      F = wrapped_import("torch.nn.functional")

    class DepthWiseConv1d(torch.nn.Module):
      def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = torch.nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

      def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

    model = DepthWiseConv1d(n_in, n_out, [3], (0, 0))
    return model(inputs)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x)


def test_dummy_input_shape():
  n_in, n_out = 11, 11
  n_batch, n_time = 3, 999

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    model = torch.nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=5, stride=5)
    x = inputs
    for i in range(3):
      x = model(x)
    return x

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_in, n_time)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, returnn_dummy_input_shape=x.shape)


def test_returnn_config_serialization():
  n_batch, n_time, n_feat = 1, 5, 7

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    arange = torch.arange(inputs.shape[2])
    return inputs + torch.reshape(arange, (1, 1, -1))

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time, n_feat)).astype("float32")
  converter = verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
    "shape": (None, n_feat), "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": 2})

  cfg = converter.get_returnn_config_serialized()
  from returnn_helpers import config_net_dict_via_serialized, dummy_run_net
  config, net_dict = config_net_dict_via_serialized(cfg)
  dummy_run_net(config)


if __name__ == "__main__":
  if len(sys.argv) <= 1:
    for k, v in sorted(globals().items()):
      if k.startswith("test_"):
        print("-" * 40)
        print("Executing: %s" % k)
        try:
          v()
        except unittest.SkipTest as exc:
          print("SkipTest:", exc)
        print("-" * 40)
    print("Finished all tests.")
  else:
    assert len(sys.argv) >= 2
    for arg in sys.argv[1:]:
      print("Executing: %s" % arg)
      if arg in globals():
        globals()[arg]()  # assume function and execute
      else:
        eval(arg)  # assume Python code and execute
