"""
Helpers for RETURNN, copied from returnn_common
"""

from __future__ import annotations
from typing import Dict, Tuple, Any
import returnn.tf.engine
import returnn.datasets


def dummy_run_net(config: Dict[str, Any], *, train: bool = False):
  """
  Runs a couple of training iterations using some dummy dataset on the net dict.
  Use this to validate that the net dict is sane.
  Note that this is somewhat slow. The whole TF session setup and net construction can take 5-30 secs or so.
  It is not recommended to use this for every single test case.

  The dummy dataset might change at some point...

  Maybe this gets extended...
  """
  from returnn.tf.engine import Engine
  from returnn.datasets import init_dataset
  from returnn.config import Config
  extern_data_opts = config["extern_data"]
  n_data_dim = extern_data_opts["data"]["dim_tags"][-1].dimension
  n_classes_dim = extern_data_opts["classes"]["sparse_dim"].dimension if "classes" in extern_data_opts else 7
  config = Config({
    "train": {
      "class": "DummyDataset", "input_dim": n_data_dim, "output_dim": n_classes_dim,
      "num_seqs": 2, "seq_len": 5},
    "debug_print_layer_output_template": True,
    "task": "train",  # anyway, to random init the net
    **config
  })
  dataset = init_dataset(config.typed_value("train"))
  engine = Engine(config=config)
  engine.init_train_from_config(train_data=dataset)
  if train:
    engine.train()
  else:
    _dummy_forward_net_returnn(engine=engine, dataset=dataset)
  return engine


def _dummy_forward_net_returnn(*, engine: returnn.tf.engine.Engine, dataset: returnn.datasets.Dataset):
  from returnn.tf.engine import Runner

  def _extra_fetches_cb(*_args, **_kwargs):
    pass  # just ignore

  output = engine.network.get_default_output_layer().output
  batches = dataset.generate_batches(
    recurrent_net=engine.network.recurrent,
    batch_size=engine.batch_size,
    max_seqs=engine.max_seqs,
    used_data_keys=engine.network.get_used_data_keys())
  extra_fetches = {
    'output': output.placeholder,
    "seq_tag": engine.network.get_seq_tags(),
  }
  for i, seq_len in output.size_placeholder.items():
    extra_fetches["seq_len_%i" % i] = seq_len
  forwarder = Runner(
    engine=engine, dataset=dataset, batches=batches,
    train=False, eval=False,
    extra_fetches=extra_fetches,
    extra_fetches_callback=_extra_fetches_cb)
  forwarder.run(report_prefix=engine.get_epoch_str() + " forward")


def config_net_dict_via_serialized(config_code: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """
  :param str config_code: via get_returnn_config_serialized
  """
  print(config_code)
  scope = {}
  exec(config_code, scope, scope)
  for tmp in ["__builtins__", "Dim", "batch_dim", "FeatureDim", "SpatialDim"]:
    scope.pop(tmp)
  config = scope
  net_dict = config["network"]
  return config, net_dict
