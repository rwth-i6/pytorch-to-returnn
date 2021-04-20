
from __future__ import annotations
from typing import Optional
from returnn.config import Config
from returnn.tf.network import TFNetwork, ExternData
from returnn.tf.layers.basic import LayerBase, SubnetworkLayer
from . import tensor as _tensor


class ReturnnContext:
  def __init__(self, *, parent: Optional[ReturnnContext] = None, name: Optional[str] = None, returnn_train_flag: bool):
    self.parent = parent
    if parent:
      assert name
      self.config = parent.config
      self.tf_name_scope = parent.network.get_absolute_name_scope_prefix() + LayerBase.cls_get_tf_scope_name(name)
      assert parent.network.extern_data.data
      self.sub_net_layer = parent.network.construct_layer(
        name=name,
        # This is just a placeholder, will be replaced in define_output.
        net_dict={name: {"class": "subnetwork", "from": "data", "subnetwork": {"output": {"class": "copy"}}}})
      assert isinstance(self.sub_net_layer, SubnetworkLayer)
      if name.startswith("."):  # temp sub net
        # We do not want that the parent net finds it.
        parent.network.layers.pop(name)
        parent.network.layers.pop(f"{name}/output")
      self._dummy_sub_output = self.sub_net_layer.subnetwork.layers["output"]
    else:
      self.config = Config({
        # "debug_print_layer_output_template": True,
      })
      self.tf_name_scope = ""
      self.sub_net_layer = None
      self._dummy_sub_output = None
    if self.sub_net_layer:
      self.network = self.sub_net_layer.subnetwork
    else:
      assert not parent
      self.network = TFNetwork(
        extern_data=ExternData(), config=self.config, name="root",
        train_flag=returnn_train_flag,
        absolute_name_prefix=(self.tf_name_scope + "/") if self.tf_name_scope else "")

  def __repr__(self):
    return f"<{self.__class__.__name__} {self.network.get_absolute_name_prefix()!r}>"

  def define_input(self, input: _tensor.TensorEntry, *, data_key: Optional[str] = None):
    if self._dummy_sub_output:
      assert self.network.layers["output"] is self._dummy_sub_output
      self._dummy_sub_output = None
      # Reset both, as we refill them. They contain dummy data.
      self.network.layers.clear()
      self.network.extern_data.data.clear()
    if data_key is None:
      data_key = self.network.extern_data.default_input
    assert data_key not in self.network.extern_data.data
    assert input.returnn_data
    self.network.extern_data.data[data_key] = input.returnn_data
    self.network.extern_data.init_batch_info()

  def define_output(self, output_layer: LayerBase):
    assert self.network.layers["output"] is output_layer
    if self.sub_net_layer:
      self.sub_net_layer.output = output_layer.output
