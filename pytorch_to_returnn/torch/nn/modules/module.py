
from __future__ import annotations
import tensorflow as tf
import numpy
from collections import OrderedDict
from typing import Optional, Callable, TypeVar, Iterator, Tuple, List, Union, Dict, Any, overload
import types
import itertools
from ..parameter import Parameter
from ...tensor import Tensor
from ...autograd import no_grad
from ...utils.hooks import RemovableHandle
from ....naming import Naming, CallEntry, TensorEntry
from returnn.tf.layers.basic import LayerBase
from returnn.tf.util.data import Data

# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar('T', bound='Module')


class Module:
  """
  Base class.

  Note::

  When porting over a PyTorch module to here, you should keep the module mostly as-is,
  i.e. mostly use the original code, and keep all parameters, buffers, etc.
  You can even keep the `forward` function as long as all other modules/functions it uses are wrapped.
  Or you remove the `forward` function, and implement `create_returnn_layer_dict`.

  Low-level modules which would wrap directly to a corresponding RETURNN layer,
  should do that, i.e. have no `forward`, but implement `create_returnn_layer_dict`.

  Other modules should work just as-is.
  I.e. this can be used as base class for external PyTorch code.

  This would wrap all the standard PyTorch modules (e.g. torch.nn.Conv1d)
  (in that case, `is_original_torch_module = True`).
  This would also be used to implement new modules
  which are needed to wrap other functions (e.g. torch.nn.functional)
  (in that case, `is_original_torch_module = False`).
  """
  # All derived classes here, which exist in PyTorch as well (e.g. torch.nn.Conv1d etc),
  # should have this set to True.
  is_original_torch_module: bool = True

  _wrapped_class_cache = {}  # cls -> WrappedClass

  # Need to overwrite to wrap __init__ to correctly set context.
  def __new__(cls, *args, **kwargs):
    if cls not in cls._wrapped_class_cache:
      class WrappedClass(cls):
        def __init__(self, *args, **kwargs):
          self.__class__ = cls  # we don't need this wrapper class anymore
          with Naming.get_instance().push_module_creation(self):
            cls.__init__(self, *args, **kwargs)
      WrappedClass.__name__ = cls.__name__
      WrappedClass.__qualname__ = cls.__qualname__
      WrappedClass.__module__ = cls.__module__
      wrapped_cls = WrappedClass
      cls._wrapped_class_cache[cls] = wrapped_cls
    else:
      wrapped_cls = cls._wrapped_class_cache[cls]
    return super(Module, cls).__new__(wrapped_cls)

  def __init__(self):
    self._parameters = OrderedDict()  # type: OrderedDict[str, Parameter]
    self._modules = OrderedDict()  # type: OrderedDict[str, Optional[Module]]
    self._buffers = OrderedDict()  # type: OrderedDict[str, Tensor]
    self._non_persistent_buffers_set = set()
    self._forward_pre_hooks = OrderedDict()

  def __repr__(self):
    return f"<{self.__class__.__name__}>"

  def get_returnn_name(self) -> str:
    return self.__class__.__name__

  # Overwrite to catch all access, to be able to wrap member functions.
  def __getattribute__(self, item: str):
    if item in {"__getattr__", "__setattr__", "__delattr__", "__dir__", "__dict__", "__class__"}:
      # Fast path, and no need for wrapping.
      return super(Module, self).__getattribute__(item)
    if hasattr(Module, item) and getattr(self.__class__, item) is getattr(Module, item):
      # Some member of the base class.
      # No wrapping needed.
      return super(Module, self).__getattribute__(item)
    obj = super(Module, self).__getattribute__(item)
    if isinstance(obj, types.MethodType):
      def wrapped_func(*args, **kwargs):
        with Naming.get_instance().push_module_context(self):
          return obj(*args, **kwargs)
      wrapped_func.__name__ = obj.__name__
      wrapped_func.__qualname__ = obj.__qualname__
      return wrapped_func
    return obj

  def __getattr__(self, item):
    if item in {"_parameters", "_modules", "_buffers"}:
      raise AttributeError(f"Module.__init__ not yet called, attrib {item!r} not present")
    if item in self._parameters:
      return self._parameters[item]
    if item in self._buffers:
      return self._buffers[item]
    if item in self._modules:
      return self._modules[item]
    raise AttributeError(f"No attrib {item!r}")

  def __setattr__(self, name, value):
    def remove_from(*dicts_or_sets):
      for d in dicts_or_sets:
        if name in d:
          if isinstance(d, dict):
            del d[name]
          else:
            d.discard(name)

    params = self.__dict__.get('_parameters')
    if isinstance(value, Parameter):
      if params is None:
        raise AttributeError("cannot assign parameters before Module.__init__() call")
      remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
      self.register_parameter(name, value)
      return

    if params is not None and name in params:
      if value is not None:
        raise TypeError(f"cannot assign {type(value)} as parameter {name!r} (torch.nn.Parameter or None expected)")
      self.register_parameter(name, value)
      return

    modules = self.__dict__.get('_modules')
    if isinstance(value, Module):
      if modules is None:
        raise AttributeError("cannot assign module before Module.__init__() call")
      remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
      self.add_module(name, value)
      return

    if modules is not None and name in modules:
      if value is not None:
        raise TypeError(f"cannot assign {type(value)} as child module {name!r} (torch.nn.Module or None expected)")
      modules[name] = value
      return

    buffers = self.__dict__.get('_buffers')
    if buffers is not None and name in buffers:
      if value is not None and not isinstance(value, Tensor):
        raise TypeError(f"cannot assign {type(value)} as buffer {name!r} (torch.Tensor or None expected)")
      buffers[name] = value
      if value is not None:
        Naming.get_instance().register_module_child_attr(self, name, value)
      return

    object.__setattr__(self, name, value)
    if isinstance(value, Tensor):
      Naming.get_instance().register_module_child_attr(self, name, value)

  def __delattr__(self, item):
    if item in self._parameters:
      del self._parameters[item]
      return
    if item in self._buffers:
      del self._buffers[item]
      return
    if item in self._modules:
      del self._modules[item]
      return
    super(Module, self).__delattr__(item)

  def __dir__(self):
    module_attrs = dir(self.__class__)
    attrs = list(self.__dict__.keys())
    parameters = list(self._parameters.keys())
    modules = list(self._modules.keys())
    buffers = list(self._buffers.keys())
    keys = module_attrs + attrs + parameters + modules + buffers
    keys = [key for key in keys if not key[0].isdigit()]  # Eliminate attrs that are not legal Python variable names
    return sorted(keys)

  def children(self) -> Iterator['Module']:
    for name, module in self.named_children():
      yield module

  def named_children(self) -> Iterator[Tuple[str, 'Module']]:
    memo = set()
    for name, module in self._modules.items():
      if module is not None and module not in memo:
        memo.add(module)
        yield name, module

  def add_module(self, name: str, module: Optional['Module']) -> None:
    if not isinstance(module, Module) and module is not None:
      raise TypeError(f"{type(module)} is not a Module subclass")
    elif not isinstance(name, str):
      raise TypeError(f"module name should be a string. Got {type(name)}")
    elif hasattr(self, name) and name not in self._modules:
      raise KeyError(f"attribute {name!r} already exists")
    elif '.' in name:
      raise KeyError("module name can't contain \".\"")
    elif name == '':
      raise KeyError("module name can't be empty string \"\"")
    self._modules[name] = module
    if module:
      Naming.get_instance().register_module_child_attr(self, name, module)

  def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
    if '_parameters' not in self.__dict__:
      raise AttributeError("cannot assign parameter before Module.__init__() call")
    elif not isinstance(name, str):
      raise TypeError(f"parameter name should be a string. Got {type(name)}")
    elif '.' in name:
      raise KeyError("parameter name can't contain \".\"")
    elif name == '':
      raise KeyError("parameter name can't be empty string \"\"")
    elif hasattr(self, name) and name not in self._parameters:
      raise KeyError("attribute '{}' already exists".format(name))
    self._parameters[name] = param
    if param is not None:
      Naming.get_instance().register_module_child_attr(self, name, param)

  def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
    if '_buffers' not in self.__dict__:
      raise AttributeError("cannot assign buffer before Module.__init__() call")
    elif not isinstance(name, str):
      raise TypeError(f"buffer name should be a string. Got {type(name)}")
    elif '.' in name:
      raise KeyError("buffer name can't contain \".\"")
    elif name == '':
      raise KeyError("buffer name can't be empty string \"\"")
    elif hasattr(self, name) and name not in self._buffers:
      raise KeyError("attribute '{}' already exists".format(name))
    elif tensor is not None and not isinstance(tensor, Tensor):
      raise TypeError(f"cannot assign {type(tensor)} object to buffer {name!r}")
    else:
      self._buffers[name] = tensor
      if persistent:
        self._non_persistent_buffers_set.discard(name)
      else:
        self._non_persistent_buffers_set.add(name)
      if tensor is not None:
        Naming.get_instance().register_module_child_attr(self, name, tensor)

  def apply(self: T, fn: Callable[['Module'], None]) -> T:
    with Naming.get_instance().push_module_apply(self):
      for module in self.children():
        module.apply(fn)
      fn(self)
    return self

  def register_forward_pre_hook(self, hook: Callable[..., None]) -> RemovableHandle:
    handle = RemovableHandle(self._forward_pre_hooks)
    self._forward_pre_hooks[handle.id] = hook
    return handle

  def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs):
    import torch
    persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
    local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}

    for name, param in local_state.items():
      key = prefix + name
      if key in state_dict:
        input_param = state_dict[key]
        if isinstance(input_param, torch.Tensor):  # we allow this here
          input_param = Tensor(*input_param.shape, numpy_array=input_param.detach().cpu().numpy())

        # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
        if len(param.shape) == 0 and len(input_param.shape) == 1:
          input_param = input_param[0]

        if input_param.shape != param.shape:
          # local shape should match the one in checkpoint
          error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                            'the shape in current model is {}.'
                            .format(key, input_param.shape, param.shape))
          continue

        try:
          with no_grad():
            param.copy_(input_param)
        except Exception as ex:
          error_msgs.append('While copying the parameter named "{}", '
                            'whose dimensions in the model are {} and '
                            'whose dimensions in the checkpoint are {}, '
                            'an exception occurred : {}.'
                            .format(key, param.size(), input_param.size(), ex.args))
      elif strict:
        missing_keys.append(key)

    if strict:
      for key in state_dict.keys():
        if key.startswith(prefix):
          input_name = key[len(prefix):]
          input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
          if input_name not in self._modules and input_name not in local_state:
            unexpected_keys.append(key)

  def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]], strict: bool = True):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
      state_dict._metadata = metadata

    def load(module, prefix=''):
      local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
      module._load_from_state_dict(
        state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
      for name, child in module._modules.items():
        if child is not None:
          load(child, prefix + name + '.')

    load(self)
    load = None  # break load->load reference cycle

    if strict:
      if len(unexpected_keys) > 0:
        error_msgs.insert(
          0, 'Unexpected key(s) in state_dict: {}. '.format(
            ', '.join('"{}"'.format(k) for k in unexpected_keys)))
      if len(missing_keys) > 0:
        error_msgs.insert(
          0, 'Missing key(s) in state_dict: {}. '.format(
            ', '.join('"{}"'.format(k) for k in missing_keys)))

    if len(error_msgs) > 0:
      raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
        self.__class__.__name__, "\n\t".join(error_msgs)))

  def _named_members(self, get_members_fn, prefix='', recurse=True):
    memo = set()
    modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
    for module_prefix, module in modules:
      members = get_members_fn(module)
      for k, v in members:
        if v is None or v in memo:
          continue
        memo.add(v)
        name = module_prefix + ('.' if module_prefix else '') + k
        yield name, v

  def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
    for name, param in self.named_parameters(recurse=recurse):
      yield param

  def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
    gen = self._named_members(
      lambda module: module._parameters.items(),
      prefix=prefix, recurse=recurse)
    for elem in gen:
      yield elem

  def eval(self):
    return self  # ignore

  def train(self, arg):
    return self  # ignore

  def to(self, *args):
    return self  # ignore

  def __call__(self, *input: Tensor, **kwargs):
    assert not kwargs  # not implemented yet
    naming = Naming.get_instance()
    with naming.push_module_context(self):
      assert naming.wrap_to_returnn_enabled
      for hook in self._forward_pre_hooks.values():
        result = hook(self, input)
        if result is not None:
          if not isinstance(result, tuple):
            result = (result,)
          input = result
      with naming.push_module_call(module=self, inputs=list(input)) as call_entry:
        res = call_entry.apply_call()
      return res

  def forward(self, *inputs: Tensor) -> Tensor:
    raise Exception("should not get here")

  def create_returnn_layer_dict(self, *inputs: Tensor) -> Dict[str, Any]:
    raise Exception("should not get here")

  @classmethod
  def has_torch_forward(cls) -> bool:
    if cls.create_returnn_layer_dict is Module.create_returnn_layer_dict:
      return True  # always assume that the user module has custom forward code, even if not cls.forward
    return cls.forward is not Module.forward

  def check_returnn_layer(self, layer: LayerBase):
    """
    You can override this function to perform extra checks on the constructed RETURNN layer,
    e.g. the right input dimension.

    :param LayerBase layer: the constructed layer
    """

  def import_params_torch_to_returnn(self, *, layer: LayerBase, torch_module):
    pass

  def check_call_returnn_outputs_to_prev_torch(self, call: CallEntry):
    naming = Naming.get_instance()
    assert naming.wrap_to_returnn_enabled and naming.import_params_from_torch_namespace and call.returnn_layer
    assert call.module.module is self
    torch_naming = naming.import_params_from_torch_namespace
    mod_entry = call.module
    mod = mod_entry.module
    call_idx = naming.get_module_call_idx(module=mod, call=call)
    mod_abs_name = naming.get_module_abs_name(mod)
    torch_mod = torch_naming.get_module_by_abs_name(mod_abs_name)
    torch_mod_calls = torch_naming.get_module_calls(torch_mod)
    assert call_idx < len(torch_mod_calls)
    torch_mod_call = torch_mod_calls[call_idx]
    assert torch_mod_call.orig_outputs is not None and torch_mod_call.orig_inputs is not None
    assert len(call.inputs) == len(torch_mod_call.orig_inputs)
    feed_dict = {}
    for tensor_entry, torch_input in zip(call.inputs, torch_mod_call.orig_inputs):
      assert isinstance(tensor_entry, TensorEntry)
      torch_axis_from_returnn_axis = {i: j for (j, i) in tensor_entry.returnn_axis_from_torch_axis.items()}
      torch_input_np = torch_input.detach().cpu().numpy()
      assert isinstance(torch_input_np, numpy.ndarray)
      assert len(torch_input_np.shape) == tensor_entry.returnn_data.batch_ndim
      torch_input_np = torch_input_np.transpose(*[torch_axis_from_returnn_axis[i] for i in range(torch_input_np.ndim)])
      feed_dict[tensor_entry.returnn_data.placeholder] = torch_input_np
      for i, size in tensor_entry.returnn_data.size_placeholder.items():
        axis = tensor_entry.returnn_data.get_batch_axis(i)
        n_batch = torch_input_np.shape[tensor_entry.returnn_data.batch_dim_axis]
        n_time = torch_input_np.shape[axis]
        size_np = [n_time] * n_batch  # fake, not so important
        feed_dict[size] = size_np
    session = tf.compat.v1.get_default_session()
    assert len(call.outputs) == 1
    returnn_output_tensor_entry = call.outputs[0]
    returnn_output_np_ = session.run(call.returnn_layer.output.placeholder, feed_dict=feed_dict)
    assert isinstance(returnn_output_np_, numpy.ndarray)
    returnn_output_np = returnn_output_np_.transpose(*[
      returnn_output_tensor_entry.returnn_axis_from_torch_axis[i] for i in range(returnn_output_np_.ndim)])
    torch_outputs = torch_mod_call.orig_outputs
    if not isinstance(torch_mod_call.orig_outputs, (list, tuple)):
      torch_outputs = [torch_outputs]
    assert len(torch_outputs) == 1
    torch_out_np = torch_outputs[0].detach().cpu().numpy()
    error_msg_info = [f"RETURNN layer: {call.returnn_layer}", f"Torch module: {torch_mod}"]
    if returnn_output_np.shape != torch_out_np.shape:
      error_msg_info += [
        "ERROR: Output shape mismatch",
        f"  RETURNN output data: {call.returnn_layer.output}",
        f"  RETURNN output shape: {returnn_output_np_.shape},",
        f"  RETURNN output shape (transposed to Torch): {returnn_output_np.shape},"
        f" (RETURNN<-Torch axis mapping {returnn_output_tensor_entry.returnn_axis_from_torch_axis})",
        f"  Torch output shape: {torch_out_np.shape}"]
      for i, (tensor_entry, torch_input) in enumerate(zip(call.inputs, torch_mod_call.orig_inputs)):
        assert isinstance(tensor_entry, TensorEntry)
        error_msg_info += [f"input {i + 1}/{len(call.inputs)}:"]
        torch_axis_from_returnn_axis = {i: j for (j, i) in tensor_entry.returnn_axis_from_torch_axis.items()}
        torch_input_np_ = torch_input.detach().cpu().numpy()
        assert isinstance(torch_input_np_, numpy.ndarray)
        assert len(torch_input_np_.shape) == tensor_entry.returnn_data.batch_ndim
        torch_input_np = torch_input_np_.transpose(
          *[torch_axis_from_returnn_axis[i] for i in range(torch_input_np_.ndim)])
        error_msg_info += [
          f"  RETURNN input data: {tensor_entry.returnn_data}",
          f"  RETURNN input shape (Torch axis order): {torch_input_np_.shape}",
          f"  RETURNN input shape (transposed to RETURNN): {torch_input_np.shape}"
          f" (Torch<-RETURNN axis mapping {torch_axis_from_returnn_axis})"]
    else:
      is_close_arr = numpy.isclose(returnn_output_np, torch_out_np, rtol=0, atol=1e-4)
      if not numpy.all(is_close_arr):
        idx = numpy.argmax(numpy.abs(returnn_output_np - torch_out_np))
        idx_ = numpy.unravel_index(idx, shape=returnn_output_np.shape)
        error_msg_info += [
          f"  RETURNN output min/max: {numpy.min(returnn_output_np), numpy.max(returnn_output_np)}",
          f"  Torch output min/max: {numpy.min(torch_out_np), numpy.max(torch_out_np)}",
          f"  Biggest mismatch at idx {idx_}, RETURNN {returnn_output_np[idx_]} vs Torch {torch_out_np[idx_]}",
        ]
    numpy.testing.assert_allclose(
      returnn_output_np, torch_out_np, rtol=0, atol=1e-4,
      err_msg="\n".join(error_msg_info))

  def _get_input_layer_name(self, input: Tensor) -> str:
    naming = Naming.get_instance()
    assert naming.module_call_stack
    top_call_entry = naming.module_call_stack[-1]
    assert top_call_entry.module.module is self
    parent_namespace = top_call_entry.namespace.parent
    # Note: If name_for_tensor fails, it means the tensor was not registered properly.
    return parent_namespace.name_for_tensor(naming.tensors[input])

  def _get_input_axis_to_returnn(self, input: Tensor, axis: int) -> str:
    return Naming.get_instance().register_tensor(input).get_returnn_axis_description(axis)

  def make_output_tensor_from_returnn(self, inputs: Tuple[Tensor, ...], layer: LayerBase) -> Tensor:
    naming = Naming.get_instance()
    torch_shape, returnn_axis_from_torch_axis = self._get_output_shape_from_returnn(inputs=inputs, layer=layer)
    is_const = False
    numpy_array = None
    inputs_entries = [naming.tensors[x] if x is not None else None for x in inputs]  # type: List[Optional[TensorEntry]]
    if all([x.is_const if x else True for x in inputs_entries]):
      # Only have const input.
      # Evaluate layer, because this const might be used in certain operation (e.g. predefined filter for conv).
      session = tf.compat.v1.get_default_session()
      feed_dict = {}
      for x in inputs_entries:
        if not x:
          continue
        feed_dict[x.returnn_data.placeholder] = x.tensor().numpy()
      numpy_array = session.run(layer.output.placeholder, feed_dict=feed_dict)
      is_const = True
    tensor = Tensor(*torch_shape, numpy_array=numpy_array, dtype=layer.output.dtype)
    tensor_entry = naming.register_tensor(tensor)
    tensor_entry.is_const = is_const
    tensor_entry.returnn_data = layer.output
    tensor_entry.returnn_axis_from_torch_axis = returnn_axis_from_torch_axis
    return tensor

  @staticmethod
  def _base_get_output_shape_from_returnn(inputs: Tuple[Tensor, ...], layer: LayerBase
                                          ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    """
    :return: (torch_shape, returnn_axis_from_torch_axis).
      Torch shape how it would have looked when this would be processed within Torch.
      The RETURNN layer.output shape (order of axes) might look different.

    This is a bit tricky.
    If axes got reordered (RETURNN does that for efficiency, and then returns as-is),
    Torch code would not expect this.
    We need a mapping of RETURNN axis -> Torch axis.
    We can automatically infer this, but this is a bit involved.
    """
    # We could also use functions like Data.get_batch_shape_dim_tags, Data.get_common_data,
    # DimensionTag.get_all_dimension_tags, etc.
    # However, this is simpler, and also more consistent with get_returnn_axis_description.
    def _get_shape_meta(data: Data) -> List[str]:
      _res = []
      for i in range(data.batch_ndim):
        if i == data.batch_dim_axis:
          _res.append("B")
        elif i == data.feature_dim_axis:
          _res.append("F")
        elif i in data.get_spatial_batch_axes():
          _res.append(f"spatial:{data.get_spatial_batch_axes().index(i)}")
        else:
          raise Exception(f"not expected {data}, axis {i}")
      return _res

    layer_output_shape_meta = _get_shape_meta(layer.output)  # e.g. [T_out,B,D_out]
    out_returnn_axis_to_torch_axis = {}
    # Torch would maybe have operated on [B,D_in,T_in] input, and produce [B,D_out,T_out] output.
    naming = Naming.get_instance()
    batch_size = None
    dyn_size = None  # currently only a single supported
    for input in inputs:
      x = naming.tensors[input]
      assert isinstance(x, TensorEntry)
      assert x.returnn_data and x.returnn_axis_from_torch_axis is not None
      if x.returnn_data.have_batch_axis():
        batch_size = input.shape[x.returnn_axis_from_torch_axis[x.returnn_data.batch_dim_axis]]
      if x.returnn_data.get_dynamic_axes():
        i = x.returnn_data.get_dynamic_axes()[0]
        dyn_size = input.shape[x.returnn_axis_from_torch_axis[i]]
      shape_meta = _get_shape_meta(x.returnn_data)  # e.g. [B,T_in,D_in], RETURNN order
      # Reorder dim tags as the input like it would look like for Torch, e.g. [B,D_in,T_in].
      shape_meta_torch_order = [shape_meta[x.returnn_axis_from_torch_axis[i]] for i in range(x.returnn_data.batch_ndim)]
      for kind in ["B", "F"]:
        if kind in shape_meta_torch_order and kind in layer_output_shape_meta:
          if shape_meta_torch_order.index(kind) == len(shape_meta_torch_order) - 1:
            out_returnn_axis_to_torch_axis[layer_output_shape_meta.index(kind)] = layer.output.batch_ndim - 1
          else:
            out_returnn_axis_to_torch_axis[layer_output_shape_meta.index(kind)] = shape_meta_torch_order.index(kind)
      break
    assert all(0 <= d < layer.output.batch_ndim for d in out_returnn_axis_to_torch_axis.values())
    assert len(set(out_returnn_axis_to_torch_axis.values())) == len(out_returnn_axis_to_torch_axis)
    rem_torch_axes = set(range(layer.output.batch_ndim)).difference(set(out_returnn_axis_to_torch_axis.values()))
    rem_returnn_axes = set(range(layer.output.batch_ndim)).difference(set(out_returnn_axis_to_torch_axis.keys()))
    assert len(rem_torch_axes) == len(rem_returnn_axes)
    for i, j in zip(sorted(rem_returnn_axes), sorted(rem_torch_axes)):
      assert i not in out_returnn_axis_to_torch_axis
      out_returnn_axis_to_torch_axis[i] = j
    assert (
        len(set(out_returnn_axis_to_torch_axis.values())) ==
        len(set(out_returnn_axis_to_torch_axis.keys())) ==
        layer.output.batch_ndim)
    out_shape = list(layer.output.batch_shape)
    if layer.output.have_batch_axis():
      assert batch_size is not None
      out_shape[layer.output.batch_dim_axis] = batch_size
    if layer.output.get_dynamic_axes():
      assert dyn_size is not None
      assert len(layer.output.get_dynamic_axes()) == 1  # not implemented otherwise
      out_shape[layer.output.get_dynamic_axes()[0]] = dyn_size
    assert all(d for d in out_shape)
    torch_axis_to_returnn = {i: j for (j, i) in out_returnn_axis_to_torch_axis.items()}
    assert len(torch_axis_to_returnn) == layer.output.batch_ndim
    out_shape = [out_shape[torch_axis_to_returnn[i]] for i in range(layer.output.batch_ndim)]
    out_returnn_axis_from_torch_axis = torch_axis_to_returnn
    return tuple(out_shape), out_returnn_axis_from_torch_axis

  def _get_output_shape_from_returnn(self,
                                     inputs: Tuple[Tensor, ...], layer: LayerBase
                                     ) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    """
    :return: (torch_shape, returnn_axis_from_torch_axis).
      Torch shape how it would have looked when this would be processed within Torch.
      The RETURNN layer.output shape (order of axes) might look different.

    Note about returnn_axis_from_torch_axis:
    This is when you would use `tf.transpose` or `numpy.transpose`,
    and you can think of `returnn_axis_from_torch_axis` as the `perm` argument, as a list.
    The dict is like {torch_axis: returnn_axis}, i.e. Torch axis -> RETURNN axis.
    You would use this if you want to transpose RETURNN axes **to** Torch axes
    (i.e. the other way around).

    This is a bit tricky.
    If axes got reordered (RETURNN does that for efficiency, and then returns as-is),
    Torch code would not expect this.
    We need a mapping of RETURNN axis -> Torch axis.
    We can automatically infer this, but this is a bit involved.
    """
    return self._base_get_output_shape_from_returnn(inputs=inputs, layer=layer)
