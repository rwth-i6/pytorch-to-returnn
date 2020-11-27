
import _setup_test_env  # noqa
import sys
import unittest
import typing
import numpy
from returnn.tf.network import TFNetwork, ExternData
from returnn.tf.layers.basic import InternalLayer
from returnn.tf.util.data import Data
from pytorch_to_returnn import torch
from pytorch_to_returnn.naming import Naming


def test_base_get_output_shape_from_returnn():
  with Naming.make_instance() as naming:
    assert isinstance(naming, Naming)
    x = torch.Tensor(64, 1, 11, 13)
    x_ = naming.register_tensor(x)
    x_.returnn_data = Data(name="x", shape=(1, 11, 13), feature_dim_axis=1)
    x_.returnn_axis_from_torch_axis = {0: 0, 1: 1, 2: 2, 3: 3}

    net = TFNetwork(extern_data=ExternData())
    # E.g. conv layer, with padding "valid", kernel size 3.
    layer = InternalLayer(name="layer", network=net, out_type={"shape": (9, 11, 32)})

    torch_shape, returnn_axis_from_torch_axis = torch.nn.Module._base_get_output_shape_from_returnn(
      inputs=(x,), layer=layer)
    assert returnn_axis_from_torch_axis == {0: 0, 1: 3, 2: 1, 3: 2}
    assert torch_shape == (64, 32, 9, 11)


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
