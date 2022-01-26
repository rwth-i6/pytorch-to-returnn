

import tensorflow as tf
import torch
import contextlib
import numpy
import types
import tempfile
from pytorch_to_returnn.pprint import pprint
from typing import Callable, Optional, Dict, Any, Tuple, List
from returnn.tf.util.data import Data, Dim, batch_dim
from pytorch_to_returnn import torch as torch_returnn
from pytorch_to_returnn.import_wrapper import wrapped_import_torch_traced, wrapped_import_torch_returnn
from pytorch_to_returnn.import_wrapper.torch_wrappers.tensor import WrappedTorchTensor
from pytorch_to_returnn.naming import Naming, TensorEntry
from pytorch_to_returnn.naming.returnn_ctx import returnn_behavior_version


ModelFuncType = Callable[[Optional[Callable[[str], types.ModuleType]], torch.Tensor], torch.Tensor]


@contextlib.contextmanager
def make_scope():
  """
  :rtype: tf.compat.v1.Session
  """
  with tf.Graph().as_default() as graph:
    with tf.compat.v1.Session(graph=graph) as session:
      yield session


class Converter:
  """
  Example code for model func::

    def model_func(wrapped_import, inputs):

        if typing.TYPE_CHECKING or not wrapped_import:
            import torch
            from parallel_wavegan import models as pwg_models
            from parallel_wavegan import layers as pwg_layers

        else:
            torch = wrapped_import("torch")
            wrapped_import("parallel_wavegan")
            pwg_models = wrapped_import("parallel_wavegan.models")
            pwg_layers = wrapped_import("parallel_wavegan.layers")

        # Initialize PWG
        pwg_config = yaml.load(open(args.pwg_config), Loader=yaml.Loader)
        pyt_device = torch.device("cpu")
        generator = pwg_models.MelGANGenerator(**pwg_config['generator_params'])
        generator.load_state_dict(
            torch.load(args.pwg_checkpoint, map_location="cpu")["model"]["generator"])
        generator.remove_weight_norm()
        pwg_model = generator.eval().to(pyt_device)
        assert pwg_config["generator_params"].get("aux_context_window", 0) == 0  # not implemented otherwise
        pwg_pqmf = pwg_layers.PQMF(pwg_config["generator_params"]["out_channels"]).to(pyt_device)

        with torch.no_grad():
            return pwg_pqmf.synthesis(pwg_model(inputs))
  """

  def __init__(self,
               model_func: ModelFuncType, *,
               inputs: numpy.ndarray,
               inputs_data_kwargs: Optional[Dict[str, Any]] = None,
               returnn_dummy_input_shape: Optional[Tuple[int]] = None,
               returnn_input_seq_lens: Optional[Dict[int, List[int]]] = None,
               use_non_wrapped_reference: bool = True,
               verify_with_torch: bool = True,
               verify_individual_model_io: bool = True,
               import_torch_params: bool = True,
               export_tf_checkpoint_save_path: Optional[str] = None,
               verify_returnn_standalone_model: bool = True,
               train: bool = False,
               ):
    """
    :param model_func:
      Gets an argument wrapped_import(str) -> module, or None. If None, should import as is.
      It also gets the inputs, converted to the right PyTorch `Tensor` object (either original or wrapped).
    :param inputs: example inputs
    """
    self._model_func = model_func
    self._inputs_np = inputs
    self._returnn_in_data_dict = self._transform_inputs_data_kwargs(inputs, inputs_data_kwargs)
    self._returnn_dummy_input_shape = returnn_dummy_input_shape
    self._returnn_input_seq_lens = returnn_input_seq_lens
    self.use_non_wrapped_reference = use_non_wrapped_reference
    self.verify_with_torch = verify_with_torch
    self.verify_individual_model_io = verify_individual_model_io
    self.import_torch_params = import_torch_params
    self.export_tf_checkpoint_save_path = export_tf_checkpoint_save_path
    self.train = train
    self._tf_checkpoint_save_path = None  # type: Optional[str]
    self.verify_returnn_standalone_model = verify_returnn_standalone_model
    self._out_ref_np = None  # type: Optional[numpy.ndarray]
    self._torch_namespace = None  # type: Optional[Naming]
    self._out_returnn_np = None  # type: Optional[numpy.ndarray]
    self._returnn_net_dict = None  # type: Optional[Dict[str, Dict[str, Any]]]
    self._non_deterministic_layer_outputs = None  # type: Optional[Dict[str, numpy.ndarray]]

  def _transform_inputs_data_kwargs(self,
                                    inputs: numpy.ndarray,
                                    inputs_data_kwargs: Optional[Dict[str, Any]] = None
                                    ) -> Dict[str, Any]:
    inputs_data_kwargs = {} if inputs_data_kwargs is None else inputs_data_kwargs.copy()
    sparse = inputs_data_kwargs.get("sparse", None) or inputs_data_kwargs.get("sparse_dim", None)
    if "dim_tags" not in inputs_data_kwargs:
      if "feature_dim_axis" not in inputs_data_kwargs and not sparse:
        assert len(inputs.shape) >= 2  # (batch,feature|channel,...)
        if inputs_data_kwargs.get("batch_dim_axis", 0) == 0:
          inputs_data_kwargs["feature_dim_axis"] = 1  # Torch uses batch-feature-major by default
      if "shape" not in inputs_data_kwargs:
        # Assume feature is static, all other dynamic.
        # Assume batch-major.
        inputs_data_kwargs["shape"] = [
          inputs.shape[i] if i == inputs_data_kwargs["feature_dim_axis"] else None
          for i in range(1, len(inputs.shape))]
    # Create dummy Data to get unified kwargs.
    data = Data(name="data", **inputs_data_kwargs)
    inputs_data_kwargs = data.get_kwargs()
    inputs_data_kwargs.pop("name")
    dim_tags = list(inputs_data_kwargs["dim_tags"])  # type: List[Dim]
    # Make some dim tags nicer.
    dim_tags = [
      d.copy(same_as_self=False, description=d.description.replace(":var-unk:", ":"))
      if ":var-unk:" in d.description
      else d
      for d in dim_tags]
    inputs_data_kwargs["dim_tags"] = dim_tags
    return inputs_data_kwargs

  def run(self):
    if self.use_non_wrapped_reference:
      self._run_reference()
    if self.verify_with_torch or self.verify_individual_model_io or self.import_torch_params:
      self._run_traced_orig_torch()
    self._run_torch_returnn_drop_in()
    if self.verify_returnn_standalone_model:
      self._run_returnn_standalone_net_dict()
      self._run_returnn_standalone_python()

  @property
  def returnn_net_dict(self) -> Dict[str, Dict[str, Any]]:
    assert self._returnn_net_dict, "Call run() first."
    return self._returnn_net_dict

  def _make_tf_feed_dict(self, input: Data, out_entry: Optional[TensorEntry] = None):
    assert input.batch_ndim == len(self._inputs_np.shape)
    assert all(input.batch_shape[i] in {None, self._inputs_np.shape[i]} for i in range(input.batch_ndim))
    n_batch = 1 if input.batch_dim_axis is None else self._inputs_np.shape[input.batch_dim_axis]
    d = {input.placeholder: self._inputs_np}
    if self._returnn_input_seq_lens is not None:
      assert tuple(self._returnn_input_seq_lens.keys()) == tuple(input.size_placeholder.keys())
      for i, size in input.size_placeholder.items():
        d[size] = self._returnn_input_seq_lens[i]
    else:
      for i, size in input.size_placeholder.items():
        d[size] = [self._inputs_np.shape[input.get_batch_axis(i)]] * n_batch  # not so relevant
    if out_entry:
      non_deterministic_tensor_entries =  []
      entries_to_visit = [out_entry]
      visited_entries = set()
      while entries_to_visit:
        entry_ = entries_to_visit.pop(0)
        if entry_ in visited_entries:
          continue
        visited_entries.add(entry_)
        if not all(call_.module.module.is_deterministic for call_ in entry_.output_from_calls):
          non_deterministic_tensor_entries.append(entry_)
          continue
        for call_ in entry_.output_from_calls:
          for prev_entry_ in call_.inputs_tensor_deps:
            if prev_entry_ not in visited_entries:
              entries_to_visit.append(prev_entry_)

      for entry in non_deterministic_tensor_entries:
        if not all(call_.module.module.is_deterministic for call_ in entry.output_from_calls):
          assert entry.validated_to_torch_tf_feed_dict is not None
          d.update(entry.validated_to_torch_tf_feed_dict)
    return d

  def _run_reference(self):
    """
    The reference, using the original import.

    This is only needed if you want to have this as the reference output,
    for further verification checks.
    Otherwise it can be skipped.
    We can use the outputs from the traced Torch variant
    (:func:`_run_traced_orig_torch`) instead.
    """
    print(">>> Running with standard reference imports...")
    torch.manual_seed(42)
    numpy.random.seed(42)
    with torch.no_grad():
      out_ref = self._model_func(None, torch.from_numpy(self._inputs_np.copy()))
      assert isinstance(out_ref, torch.Tensor)
      out_ref_np = out_ref.cpu().numpy()
    self._out_ref_np = out_ref_np
    print()

  def _run_traced_orig_torch(self):
    """
    Now with wrapped import. That will also use the original PyTorch code, but wrapped with our custom logic.
    This should not change anything, and still would use the PyTorch logic,
    except that the wrapped classes can collect additional information.
    However, we still will check that we got the same output,
    just to check that there is no subtle bug due to the wrapping logic.
    """
    print(">>> Running with wrapped imports, wrapping original PyTorch...")
    torch.manual_seed(42)
    numpy.random.seed(42)
    with torch.no_grad():
      with Naming.make_instance(
            wrap_to_returnn_enabled=False,
            keep_orig_module_io_tensors=self.verify_individual_model_io) as naming:
        wrapped_torch = wrapped_import_torch_traced("torch")
        out_wrapped = self._model_func(wrapped_import_torch_traced, wrapped_torch.from_numpy(self._inputs_np.copy()))
        assert isinstance(out_wrapped, WrappedTorchTensor)
        out_wrapped_np = out_wrapped.cpu().numpy()
        print(">>>> Module naming hierarchy:")
        naming.root_namespace.dump()
        print(">>>> Root module calls:")
        pprint(dict(naming.get_root_module_calls()))
        torch_mods_with_params = naming.get_modules_with_params_by_abs_name()
        print(">>>> Modules with params:")
        pprint(dict(torch_mods_with_params))
        self._torch_namespace = naming
    if self._out_ref_np is not None:
      assert self._out_ref_np.shape == out_wrapped_np.shape
      numpy.testing.assert_allclose(self._out_ref_np, out_wrapped_np)
      print(">>>> Looks good!")
    else:
      self._out_ref_np = out_wrapped_np  # just use that as further reference
    print()

  def _run_torch_returnn_drop_in(self):
    print(">>> Running with wrapped Torch import, wrapping replacement for PyTorch...")
    torch.manual_seed(42)
    numpy.random.seed(42)
    with make_scope() as session:
      with Naming.make_instance(
            wrap_to_returnn_enabled=True,
            returnn_train_flag=self.train,
            keep_orig_module_io_tensors=True,  # it's only symbolic anyway in TF
            import_params_from_torch_namespace=self._torch_namespace) as naming:
        assert isinstance(naming, Naming)
        in_returnn = torch_returnn.from_numpy(self._inputs_np.copy())
        assert isinstance(in_returnn, torch_returnn.Tensor)
        x = naming.register_input(in_returnn, Data("data", **self._returnn_in_data_dict))
        print("RETURNN input:", x)
        out_returnn = self._model_func(wrapped_import_torch_returnn, in_returnn)
        assert isinstance(out_returnn, torch_returnn.Tensor)
        out_returnn_ = naming.register_output(out_returnn)
        y, returnn_axis_from_torch_axis = out_returnn_.returnn_data, out_returnn_.returnn_axis_from_torch_axis
        assert isinstance(y, Data)
        print("RETURNN output:", y, "axis map RETURNN<-Torch", returnn_axis_from_torch_axis)
        print(">>>> Module naming hierarchy:")
        naming.root_namespace.dump()
        print(">>>> RETURNN net dict:")
        self._returnn_net_dict = naming.root_namespace.dump_as_returnn_net_dict()
        pprint(self._returnn_net_dict)
        print(">>>> Root module calls:")
        pprint(dict(naming.get_root_module_calls()))
        torch_mods_with_params = naming.get_modules_with_params_by_abs_name()
        print(">>>> Modules with params:")
        pprint(dict(torch_mods_with_params))
        self._non_deterministic_layer_outputs = naming.non_deterministic_layer_outputs

      feed_dict = self._make_tf_feed_dict(x, out_returnn_)
      y_, y_size = session.run((y.placeholder, y.size_placeholder.as_dict()), feed_dict=feed_dict)
      assert isinstance(y_, numpy.ndarray)
      self._out_returnn_np = y_
      print("Output shape:", y_.shape)
      print("Output seq lens:", y_size)
      y_torch = y_.transpose(*[returnn_axis_from_torch_axis[i] for i in range(y_.ndim)])
      print("Output shape (converted to Torch):", y_torch.shape)
      if self._out_ref_np is not None:
        if self._returnn_input_seq_lens is not None and y_size:
          # ignore padded elements in comparison
          assert tuple(y_size.keys()) == tuple(self._returnn_input_seq_lens.keys())
          assert self._out_ref_np.shape == y_torch.shape, "Mismatching shapes"
          for ax in y.get_spatial_axes():
            assert y_size[0].tolist() == self._returnn_input_seq_lens[0], (
              "Mismatch of sequence lens, not yet implemented")
            torch_axis_from_returnn_axis = {j: i for i, j in returnn_axis_from_torch_axis.items()}
            batch_dim_axis = torch_axis_from_returnn_axis[y.batch_dim_axis]
            spatial_axis = torch_axis_from_returnn_axis[y.get_batch_axis(ax)]
            for seq_idx, seq_len in enumerate(y_size[ax]):
              slc = [slice(None)] * len(y_torch.shape)
              slc[batch_dim_axis] = seq_idx
              slc[spatial_axis] = slice(seq_len, y_torch.shape[spatial_axis])
              self._out_ref_np[tuple(slc)] = numpy.nan
        m = ~numpy.isnan(self._out_ref_np)
        numpy.testing.assert_allclose(self._out_ref_np[m], y_torch[m], **naming.validate_allclose_kwargs)
        print(">>>> Looks good!")

      if self.export_tf_checkpoint_save_path or self.verify_returnn_standalone_model:
        returnn_net = naming.root_namespace.returnn_ctx.network
        returnn_net.print_network_info(name="RETURNN network")
        if self.export_tf_checkpoint_save_path:
          self._tf_checkpoint_save_path = self.export_tf_checkpoint_save_path
        else:
          tmp_dir = tempfile.mkdtemp("tmp-returnn-tf-checkpoint")
          self._tf_checkpoint_save_path = tmp_dir + "/model"
        print(f"Saving TF checkpoint to {self._tf_checkpoint_save_path!r}...")
        returnn_net.global_train_step.load(0, session=session)
        returnn_net.save_params_to_file(filename=self._tf_checkpoint_save_path, session=session)
        print()

  def _run_returnn_standalone_net_dict(self):
    print(">>> Constructing RETURNN model, load TF checkpoint, run...")
    with make_scope() as session:
      from returnn.config import Config
      from returnn.tf.network import TFNetwork
      config = Config({
        "extern_data": {"data": self._returnn_in_data_dict},
        "debug_print_layer_output_template": True,
        "behavior_version": returnn_behavior_version,
      })
      network = TFNetwork(config=config, name="root", train_flag=self.train)
      network.construct_from_dict(self._returnn_net_dict)
      network.load_params_from_file(filename=self._tf_checkpoint_save_path, session=session)

      x = network.extern_data.get_default_input_data()
      y = network.get_default_output_layer().output
      feed_dict = self._make_tf_feed_dict(x)
      feed_dict.update({
        network.layers[layer_name].output.placeholder: value for layer_name, value in
        self._non_deterministic_layer_outputs.items()})
      y_, y_size = session.run((y.placeholder, y.size_placeholder.as_dict()), feed_dict=feed_dict)
      assert isinstance(y_, numpy.ndarray)
      print("Output shape:", y_.shape)
      numpy.testing.assert_allclose(self._out_returnn_np, y_)
      print(">>>> Looks good!")
      print()

  def _run_returnn_standalone_python(self):
    print(">>> Constructing RETURNN model via Python code, load TF checkpoint, run...")
    with make_scope() as session:
      with Naming.make_instance() as naming:  # we expect this to work with the default settings
        model_func = self._model_func

        # Wrap the model_func in a module.
        # We assume this would be flattened away in the namespace.
        # All named modules should thus have the same names.
        class DummyModule(torch_returnn.nn.Module):
          returnn_dummy_input_shape = self._returnn_dummy_input_shape

          def get_returnn_name(self) -> str:
            return ""  # also avoid that this name becomes a prefix anywhere

          def forward(self, *inputs):
            return model_func(wrapped_import_torch_returnn, *inputs)

        dummy_mod = DummyModule()
        net_dict = dummy_mod.as_returnn_net_dict(self._returnn_in_data_dict)

      from returnn.config import Config
      from returnn.tf.network import TFNetwork
      config = Config({
        "extern_data": {"data": self._returnn_in_data_dict},
        "debug_print_layer_output_template": True,
        "behavior_version": returnn_behavior_version,
      })
      network = TFNetwork(config=config, name="root", train_flag=self.train)
      network.construct_from_dict(net_dict)
      network.load_params_from_file(filename=self._tf_checkpoint_save_path, session=session)

      x = network.extern_data.get_default_input_data()
      y = network.get_default_output_layer().output
      feed_dict = self._make_tf_feed_dict(x)
      feed_dict.update({
        network.layers[layer_name].output.placeholder: value for layer_name, value in
        self._non_deterministic_layer_outputs.items()})
      y_, y_size = session.run((y.placeholder, y.size_placeholder.as_dict()), feed_dict=feed_dict)
      assert isinstance(y_, numpy.ndarray)
      print("Output shape:", y_.shape)
      numpy.testing.assert_allclose(self._out_returnn_np, y_)
      print(">>>> Looks good!")
      print()


def verify_torch_and_convert_to_returnn(
      model_func: ModelFuncType, *,
      inputs: numpy.ndarray, **kwargs) -> Converter:
  """
  :param model_func:
    Gets an argument wrapped_import(str) -> module, or None. If None, should import as is.
    It also gets the inputs, converted to the right PyTorch `Tensor` object (either original or wrapped).
  :param inputs:

  model_func will get called multiple times, with different wrapped_import functions.
  wrapped_import would import some user model code.
  wrapped_import expects that the user model code is still unmodified,
  using the original `import torch` statements.

  See the `readme <..>`_ for further details.
  """
  converter = Converter(model_func=model_func, inputs=inputs, **kwargs)
  converter.run()
  return converter
