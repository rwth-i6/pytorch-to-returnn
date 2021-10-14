
import _setup_test_env  # noqa
import sys
import unittest
from typing import Dict, Any
import numpy
import contextlib
import tensorflow as tf
from pytorch_to_returnn.naming import Naming
from pytorch_to_returnn import torch
from pytorch_to_returnn.torch import Tensor
from pytorch_to_returnn.pprint import pformat


def run_returnn(module: torch.nn.Module, returnn_data_dict: Dict[str, Any], train_flag=False):
  with make_scope() as session:
    from returnn.config import Config
    from returnn.tf.network import TFNetwork
    config = Config({
      "extern_data": {"data": returnn_data_dict},
      "debug_print_layer_output_template": True,
    })
    net_dict = module.as_returnn_net_dict(returnn_data_dict)
    print()
    print("Net dict:")
    print(pformat(net_dict))
    network = TFNetwork(config=config, name="root", train_flag=train_flag)
    network.construct_from_dict(net_dict)
    out = network.get_default_output_layer().output
    session.run(tf.compat.v1.global_variables_initializer())
    session.run((out.placeholder, out.size_placeholder.as_dict()), feed_dict=make_feed_dict(network.extern_data))


@contextlib.contextmanager
def make_scope():
  """
  :rtype: tf.compat.v1.Session
  """
  with tf.Graph().as_default() as graph:
    with tf.compat.v1.Session(graph=graph) as session:
      yield session


def make_feed_dict(data_list, same_time=False, n_batch=3, n_time=7):
  """
  :param list[returnn.tf.util.data.Data]|ExternData data_list:
  :param bool same_time:
  :param int n_batch:
  :param int n_time:
  :rtype: dict[tf.Tensor,numpy.ndarray]
  """
  from returnn.tf.network import ExternData
  if isinstance(data_list, ExternData):
    data_list = [value for (key, value) in sorted(data_list.data.items())]
  assert n_time > 0 and n_batch > 0
  rnd = numpy.random.RandomState(42)
  existing_sizes = {}  # type: Dict[tf.Tensor,int]
  d = {}
  for data in data_list:
    shape = list(data.batch_shape)
    if data.batch_dim_axis is not None:
      shape[data.batch_dim_axis] = n_batch
    for axis, dim in enumerate(shape):
      if dim is None:
        axis_wo_b = data.get_batch_axis_excluding_batch(axis)
        assert axis_wo_b in data.size_placeholder
        dyn_size = data.size_placeholder[axis_wo_b]
        if dyn_size in existing_sizes:
          shape[axis] = existing_sizes[dyn_size]
          continue
        existing_sizes[dyn_size] = n_time
        shape[axis] = n_time
        dyn_size_v = numpy.array([n_time, max(n_time - 2, 1), max(n_time - 3, 1)])
        if dyn_size_v.shape[0] > n_batch:
          dyn_size_v = dyn_size_v[:n_batch]
        elif dyn_size_v.shape[0] < n_batch:
          dyn_size_v = numpy.concatenate(
            [dyn_size_v, rnd.randint(1, n_time + 1, size=(n_batch - dyn_size_v.shape[0],))], axis=0)
        d[dyn_size] = dyn_size_v
        if not same_time:
          n_time += 1
    print("%r %r: shape %r" % (data, data.placeholder, shape))
    if data.sparse:
      d[data.placeholder] = rnd.randint(0, data.dim or 13, size=shape, dtype=data.dtype)
    else:
      d[data.placeholder] = rnd.normal(size=shape).astype(data.dtype)
  return d


def test_Squeeze():
  class Mod(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
      print("x:", x)
      naming = Naming.get_instance()
      x_meta = naming.get_tensor(x)
      assert x_meta.returnn_axis_from_torch_axis == {0: 0, 1: 1, 2: 2}  # id, no reordering
      squeeze_mod = torch.nn.Squeeze(2)
      y = squeeze_mod(x)
      print("y:", y)
      assert y.shape == x.shape[:2]
      y_meta = naming.get_tensor(y)
      assert y_meta.returnn_axis_from_torch_axis == {0: 0, 1: 1}  # id, no reordering
      return y

  mod = Mod()
  run_returnn(module=mod, returnn_data_dict={"shape": (None, 1)})


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
