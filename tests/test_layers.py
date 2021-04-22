
import _setup_test_env  # noqa
import sys
import unittest
import typing
import numpy
import tensorflow as tf
from pytorch_to_returnn import torch
from pytorch_to_returnn.converter import verify_torch_and_convert_to_returnn
from pytorch_to_returnn.pprint import pformat


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


def test_cat():

  def model_func(wrapped_import, inputs: torch.Tensor):
      if wrapped_import:
        torch = wrapped_import("torch")
      else:
        import torch
      return torch.cat((inputs, inputs), dim=-1)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (3,3)).astype("float32")
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


def test_matmul():
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


def test_reshape():
  n_batch, n_time, n_feature_1, n_feature_2 = 3, 5, 8, 6

  def model_func_1(wrapped_import, inputs: torch.Tensor):
    # test case (..., a*b, c,...) -> (..., a, b*c,...)
    return inputs.view(n_batch, n_time, n_feature_1 // 2, n_feature_2 * 2)

  def model_func_2(wrapped_import, inputs: torch.Tensor):
    # test case (..., a, b*c,...) -> (..., a*b, c,...)
    return inputs.view(n_batch, n_time, n_feature_1 * 2, n_feature_2 // 2)

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_time, n_feature_1, n_feature_2)).astype("float32")
  for model_func in [model_func_1, model_func_2]:
    verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={
        "shape": (None, n_feature_1, n_feature_2), "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": 2})


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
        mean_returnn = tf.squeeze(call.returnn_layer.params["batch_norm/mean"]).eval()
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


def test_broadcast_add_bias():
  n_batch, n_feature, n_time = 3, 5, 17

  def model_func(wrapped_import, inputs: torch.Tensor):
    if typing.TYPE_CHECKING or not wrapped_import:
      import torch
    else:
      torch = wrapped_import("torch")
    rnd = numpy.random.RandomState(42)
    bias = rnd.normal(0., 1., (n_feature)).astype("float32")
    bias = torch.from_numpy(bias)
    return inputs + bias

  rnd = numpy.random.RandomState(42)
  x = rnd.normal(0., 1., (n_batch, n_feature)).astype("float32")
  verify_torch_and_convert_to_returnn(model_func, inputs=x, inputs_data_kwargs={"time_dim_axis": None})


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
  converter = verify_torch_and_convert_to_returnn(model_func, inputs=x)
  assert converter.returnn_net_dict["Flatten"]["class"] == "flatten_batch", pformat(converter.returnn_net_dict)


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


def test_expand():
  n_batch, n_feat = 5, 1

  def model_func(wrapped_import, inputs: torch.Tensor):
    x = inputs.expand(3, 2, -1, 7)
    return x

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
