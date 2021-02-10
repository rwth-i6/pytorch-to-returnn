
from __future__ import annotations
import tensorflow as tf
from tensorflow.python.util import nest
import numpy
from collections import OrderedDict
from typing import Optional, Callable, TypeVar, Iterator, Tuple, List, Set, Union, Dict, Any, Collection
import types
import itertools
from ..parameter import Parameter
from ...tensor import Tensor
from ...autograd import no_grad
from ...utils.hooks import RemovableHandle
from ....naming import Naming, CallEntry, TensorEntry
from returnn.tf.layers.basic import LayerBase, SubnetworkLayer
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
  (If the original `forward` returns not a single tensor, but a tuple, or some other nested structure,
   override `make_structured_returnn_output`.)

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
    if module is not None:
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

  def named_modules(self, memo: Optional[Set[Module]] = None, prefix: str = ''):
    if memo is None:
      memo = set()
    if self not in memo:
      memo.add(self)
      yield prefix, self
      for name, module in self._modules.items():
        if module is None:
          continue
        submodule_prefix = prefix + ('.' if prefix else '') + name
        for m in module.named_modules(memo, submodule_prefix):
          yield m

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

  def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
    for name, buf in self.named_buffers(recurse=recurse):
      yield buf

  def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
    gen = self._named_members(
      lambda module: module._buffers.items(),
      prefix=prefix, recurse=recurse)
    for elem in gen:
      yield elem

  def eval(self):
    return self  # ignore

  def train(self, arg):
    return self  # ignore

  def to(self, *args):
    return self  # ignore

  @property
  def training(self):
    return False

  def __call__(self, *input: Tensor, **kwargs):
    naming = Naming.get_instance()
    with naming.push_module_context(self):
      assert naming.wrap_to_returnn_enabled
      for hook in self._forward_pre_hooks.values():
        result = hook(self, input)
        if result is not None:
          if not isinstance(result, tuple):
            result = (result,)
          input = result
      with naming.push_module_call(module=self, inputs_args=input, inputs_kwargs=kwargs) as call_entry:
        res = call_entry.apply_call()
      return res

  def forward(self, *inputs: Tensor, **kwargs) -> Tensor:
    """
    Note:

    Normally we should never get here.

    However, there is one exception:
    We can still get here, in case some user code overrides some of the Torch modules, e.g. as in::

      class LayerNorm(torch.nn.LayerNorm):
          def __init__(self, nout, dim=-1):
              super(LayerNorm, self).__init__(nout, eps=1e-12)
              self.dim = dim
          def forward(self, x):
              if self.dim == -1:
                  return super(LayerNorm, self).forward(x)
              return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)

    In that case, has_torch_forward() will return True (which is correct and expected),
    and then the forward() will be called, and that will lead to a call here.
    So we are expected to be in a subnetwork.
    But now we want to get the behavior of the original Torch module.
    """
    self_ = self
    orig_import_params_torch_to_returnn = self.import_params_torch_to_returnn

    def import_params_torch_to_returnn(*, layer: SubnetworkLayer, torch_module):
      assert isinstance(layer, SubnetworkLayer)
      sub_layer = layer.subnetwork.layers[self_.get_returnn_name()]
      orig_import_params_torch_to_returnn(layer=sub_layer, torch_module=torch_module)

    setattr(self, "import_params_torch_to_returnn", import_params_torch_to_returnn)
    setattr(self, "has_torch_forward", lambda: False)

    class _WrappedBaseClass(Module):
      is_original_torch_module = False

      def create_returnn_layer_dict(self, *inputs: Tensor, **kwargs) -> Dict[str, Any]:
        return self_.create_returnn_layer_dict(*inputs, **kwargs)

      def make_structured_returnn_output(self, output):
        return self_.make_structured_returnn_output(output)

      def get_returnn_name(self) -> str:
        return self_.get_returnn_name()

    _WrappedBaseClass.__qualname__ = f"_{self.__class__.__qualname__}_WrappedBaseClass"  # doesn't matter, but nicer
    _WrappedBaseClass.__name__ = f"_{self.__class__.__name__}_WrappedBaseClass"  # doesn't matter, but nicer
    wrapped_mod = _WrappedBaseClass()
    return wrapped_mod(*inputs)

  def create_returnn_layer_dict(self, *inputs: Tensor, **kwargs) -> Dict[str, Any]:
    raise Exception("should not get here")

  def make_structured_returnn_output(self, output: Tensor) -> Union[Tensor, Tuple[Tensor], Any]:
    """
    This can be overridden alongside with `create_returnn_layer_dict`.
    In case the original `forward` would return not a single tensor but some tuple or dict or other nested structure,
    override this to match the original `forward` return structure.
    """
    return output

  @classmethod
  def has_torch_forward(cls) -> bool:
    """
    When we return True here, it implies that this module has a `forward` function,
    which would be wrapped via a RETURNN subnetwork.

    When we return False, it implies that this module has a `create_returnn_layer_dict` function,
    and would be wrapped directly via the corresponding RETURNN layer.
    """
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
    """
    Override this function to import parameters from PyTorch.
    This only makes sense for modules available in PyTorch itself,
    with flag ``is_original_torch_module=True``.
    """

  @staticmethod
  def _check_call_returnn_input_to_prev_torch(call: CallEntry, tensor: TensorEntry, torch_values: numpy.ndarray):
    naming = Naming.get_instance()
    # Search backward to resolve all deps -- either to const, input, or already checked.
    # We keep all sizes separate, because we keep them in all cases,
    # as the RETURNN size optimization might mess up the dependency order.
    feed_dict = {}
    sizes_feed_dict = {}
    visited = set()
    queue = [tensor]
    while queue:
      queue_ = []
      for t in queue:
        if not isinstance(t, TensorEntry):
          continue
        if t in visited:
          continue
        visited.add(t)
        if t.validated_to_torch:
          pass
        elif t.is_const:
          print(f"**** validate: add const tensor {t}")
          t.validated_to_torch = True  # Skip check next time.
          # No need to feed, this should be const.
          t.validated_to_torch_tf_feed_dict = {}
          t.validated_to_torch_tf_sizes_feed_dict = {}
        elif t.is_input:
          print(f"**** validate: add network input tensor {t}")
          t.validated_to_torch = True
          t.validated_to_torch_tf_feed_dict = {}
          # Should be set via register_input.
          t_ = t.tensor()
          assert isinstance(t_, Tensor)
          data = t_.numpy()
          t.validated_to_torch_tf_feed_dict[t.returnn_data.placeholder] = data
          t.validated_to_torch_tf_sizes_feed_dict = {}
          for i, size in t.returnn_data.size_placeholder.items():
            batch_dim = data.shape[t.returnn_data.batch_dim_axis]
            size_ = data.shape[t.returnn_data.get_batch_axis(i)]
            t.validated_to_torch_tf_sizes_feed_dict[size] = numpy.array([size_] * batch_dim, dtype=numpy.int32)
          t.validated_to_torch_tf_feed_dict.update(t.validated_to_torch_tf_sizes_feed_dict)
        else:
          assert len(t.output_from_calls) >= 1
          call_ = t.output_from_calls[0]
          queue_ += call_.inputs_flat
          continue

        assert t.validated_to_torch
        feed_dict.update(t.validated_to_torch_tf_feed_dict)
        sizes_feed_dict.update(t.validated_to_torch_tf_sizes_feed_dict)

      queue = queue_

    print(f"**** validate: add call {call} input tensor {tensor}")
    session = tf.compat.v1.get_default_session()
    returnn_output_np_, output_sizes = session.run(
      (tensor.returnn_data.placeholder,
       tensor.returnn_data.size_placeholder),
      feed_dict=feed_dict)
    assert isinstance(returnn_output_np_, numpy.ndarray)
    tensor.validated_to_torch = True
    tensor.validated_to_torch_tf_feed_dict = {
      tensor.returnn_data.placeholder: returnn_output_np_}
    out_size_feed_dict = {
      tensor.returnn_data.size_placeholder[i]: output_sizes[i]
      for i in tensor.returnn_data.size_placeholder}
    out_size_feed_dict.update(sizes_feed_dict)
    tensor.validated_to_torch_tf_feed_dict.update(out_size_feed_dict)
    tensor.validated_to_torch_tf_sizes_feed_dict = out_size_feed_dict

    # Check now against Torch reference.
    torch_axis_from_returnn_axis = {i: j for (j, i) in tensor.returnn_axis_from_torch_axis.items()}
    assert len(torch_values.shape) == tensor.returnn_data.batch_ndim
    torch_values_ = torch_values.transpose(*[torch_axis_from_returnn_axis[i] for i in range(torch_values.ndim)])
    numpy.testing.assert_allclose(
      returnn_output_np_, torch_values_,
      err_msg=f"call {call} input tensor {tensor} check failed",
      **naming.validate_allclose_kwargs)

  @staticmethod
  def check_call_returnn_outputs_to_prev_torch(call: CallEntry, *, update_ops: List[tf.Operation]):
    naming = Naming.get_instance()
    assert naming.wrap_to_returnn_enabled and naming.import_params_from_torch_namespace and call.returnn_layer
    torch_naming = naming.import_params_from_torch_namespace
    mod_entry = call.module
    mod = mod_entry.module
    call_idx = naming.get_module_call_idx(module=mod, call=call)
    mod_abs_name = naming.get_module_abs_id_name(mod)
    # If the following throws an exception, maybe this module was marked with is_original_torch_module=True,
    # but actually it does not exist in Torch, or would not be used like this.
    # E.g. in the functional API, any created modules should use `as_returnn_torch_functional()`.
    torch_mod = torch_naming.get_module_by_abs_id_name(mod_abs_name)
    torch_mod_calls = torch_naming.get_module_calls(torch_mod)
    assert call_idx < len(torch_mod_calls)
    torch_mod_call = torch_mod_calls[call_idx]
    assert torch_mod_call.orig_outputs is not None and torch_mod_call.orig_inputs_flat is not None
    assert len(call.inputs_flat) == len(torch_mod_call.orig_inputs_flat)
    feed_dict = {}
    sizes_feed_dict = {}
    for tensor_entry, torch_input in zip(call.inputs_flat, torch_mod_call.orig_inputs_flat):
      if not isinstance(tensor_entry, TensorEntry):
        continue
      torch_input_np = torch_input.detach().cpu().numpy()
      Module._check_call_returnn_input_to_prev_torch(
        call=call, tensor=tensor_entry, torch_values=torch_input_np)
      feed_dict.update(tensor_entry.validated_to_torch_tf_feed_dict)
      sizes_feed_dict.update(tensor_entry.validated_to_torch_tf_sizes_feed_dict)
    session = tf.compat.v1.get_default_session()
    nest.assert_same_structure(call.outputs, torch_mod_call.orig_outputs)
    assert len(call.outputs_flat) == len(torch_mod_call.orig_outputs_flat)
    for i, x in enumerate(call.outputs_flat):
      idx_repr = f" {i + 1}/{len(call.outputs_flat)}" if len(call.outputs_flat) > 1 else ""
      print(f"**** validate: add call {call} output{idx_repr} tensor {x}")
    out, _ = session.run(
      ([(x.returnn_data.placeholder, x.returnn_data.size_placeholder) for x in call.outputs_flat], update_ops),
      feed_dict=feed_dict)
    for out_idx, returnn_output_tensor_entry in enumerate(call.outputs_flat):
      idx_repr = f" {out_idx + 1}/{len(call.outputs_flat)}" if len(call.outputs_flat) > 1 else ""
      returnn_output_np_, output_sizes = out[out_idx]
      assert isinstance(returnn_output_np_, numpy.ndarray)
      returnn_output_tensor_entry.validated_to_torch = True
      returnn_output_tensor_entry.validated_to_torch_tf_feed_dict = {
        returnn_output_tensor_entry.returnn_data.placeholder: returnn_output_np_}
      out_size_feed_dict = {
        returnn_output_tensor_entry.returnn_data.size_placeholder[i]: output_sizes[i]
        for i in returnn_output_tensor_entry.returnn_data.size_placeholder}
      out_size_feed_dict.update(sizes_feed_dict)
      returnn_output_tensor_entry.validated_to_torch_tf_feed_dict.update(out_size_feed_dict)
      returnn_output_tensor_entry.validated_to_torch_tf_sizes_feed_dict = out_size_feed_dict
      returnn_output_np = returnn_output_np_.transpose(*[
        returnn_output_tensor_entry.returnn_axis_from_torch_axis[i] for i in range(returnn_output_np_.ndim)])
      torch_out_np = torch_mod_call.orig_outputs_flat[out_idx].detach().cpu().numpy()
      error_msg_info = [f"RETURNN layer: {call.returnn_layer}", f"Torch module: {torch_mod}"]
      if returnn_output_np.shape != torch_out_np.shape:
        error_msg_info += [
          f"ERROR: Output{idx_repr} shape mismatch",
          f"  RETURNN output{idx_repr} data: {call.outputs_flat[out_idx].returnn_data}",
          f"  RETURNN output{idx_repr} shape: {returnn_output_np_.shape},",
          f"  RETURNN output{idx_repr} shape (transposed to Torch): {returnn_output_np.shape},"
          f" (RETURNN<-Torch axis mapping {returnn_output_tensor_entry.returnn_axis_from_torch_axis})",
          f"  Torch output shape: {torch_out_np.shape}"]
        for i, (tensor_entry, torch_input) in enumerate(zip(call.inputs_flat, torch_mod_call.orig_inputs_flat)):
          error_msg_info += [f"input {i + 1}/{len(call.inputs_flat)}:"]
          if not isinstance(tensor_entry, TensorEntry):
            error_msg_info += ["  (non tensor)"]
            continue
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
        is_close_arr = numpy.isclose(returnn_output_np, torch_out_np, **naming.validate_allclose_kwargs)
        if not numpy.all(is_close_arr):
          idx = numpy.argmax(numpy.abs(returnn_output_np - torch_out_np))
          idx_ = numpy.unravel_index(idx, shape=returnn_output_np.shape)
          error_msg_info += [
            f"  RETURNN output{idx_repr} min/max: {numpy.min(returnn_output_np), numpy.max(returnn_output_np)}",
            f"  Torch output{idx_repr} min/max: {numpy.min(torch_out_np), numpy.max(torch_out_np)}",
            f"  Biggest mismatch at idx {idx_}, RETURNN {returnn_output_np[idx_]} vs Torch {torch_out_np[idx_]}",
          ]
      numpy.testing.assert_allclose(
        returnn_output_np, torch_out_np,
        err_msg="\n".join(error_msg_info),
        **naming.validate_allclose_kwargs)

  @staticmethod
  def _get_input_layer_name(input: Tensor) -> str:
    naming = Naming.get_instance()
    assert naming.module_call_stack
    top_call_entry = naming.module_call_stack[-1]
    parent_namespace = top_call_entry.namespace.parent
    x = naming.get_tensor(input)
    naming.prepare_tensor_as_input(x, parent_namespace=parent_namespace)
    # Note: If name_for_tensor fails, it means the tensor was not registered properly.
    name, _ = parent_namespace.name_for_tensor(x)
    return name

  @staticmethod
  def _assert_spatial_axes_in_order(input: Tensor):
    entry = Naming.get_instance().register_tensor(input)
    assert entry.returnn_data
    spatial_axes = entry.returnn_data.get_spatial_batch_axes()
    returnn_axis_to_torch_axis = {i: j for (j, i) in entry.returnn_axis_from_torch_axis.items()}
    spatial_axes_torch = [returnn_axis_to_torch_axis[i] for i in spatial_axes]
    assert sorted(spatial_axes_torch) == spatial_axes_torch

  @staticmethod
  def _assert_axes_in_order(input: Tensor, *, dims: Collection[int]):
    entry = Naming.get_instance().register_tensor(input)
    assert entry.returnn_data
    spatial_axes_returnn = [entry.returnn_axis_from_torch_axis[i] for i in sorted(dims)]
    assert sorted(spatial_axes_returnn) == spatial_axes_returnn

  @staticmethod
  def _get_input_axis_to_returnn(input: Tensor, axis: int) -> str:
    return Naming.get_instance().register_tensor(input).get_returnn_axis_description(axis)

  def make_output_tensor_from_returnn(self, inputs_flat: List[Tensor], layer: LayerBase) -> Tensor:
    naming = Naming.get_instance()
    torch_shape, returnn_axis_from_torch_axis = self._get_output_shape_from_returnn(
      inputs_flat=inputs_flat, layer=layer)
    for i in range(len(torch_shape)):
      assert layer.output.batch_shape[returnn_axis_from_torch_axis[i]] in {None, torch_shape[i]}
    is_const = False
    numpy_array = None
    inputs_entries = [
      naming.tensors[x] if isinstance(x, Tensor) else None for x in inputs_flat]  # type: List[Optional[TensorEntry]]
    if all([x.is_const if x else True for x in inputs_entries]):
      # Only have const input.
      # Evaluate layer, because this const might be used in certain operation (e.g. predefined filter for conv).
      session = tf.compat.v1.get_default_session()
      feed_dict = {}
      for x in inputs_entries:
        if not x:
          continue
        value = x.tensor().numpy()
        value = numpy.transpose(value, [x.torch_axis_from_returnn_axis[i] for i in range(value.ndim)])
        feed_dict[x.returnn_data.placeholder] = value
      numpy_array = session.run(layer.output.placeholder, feed_dict=feed_dict)
      numpy_array = numpy.transpose(numpy_array, [returnn_axis_from_torch_axis[i] for i in range(numpy_array.ndim)])
      is_const = True
    tensor = Tensor(*torch_shape, numpy_array=numpy_array, dtype=layer.output.dtype)
    tensor_entry = naming.register_tensor(tensor)
    tensor_entry.is_const = is_const
    tensor_entry.returnn_data = layer.output
    tensor_entry.returnn_axis_from_torch_axis = returnn_axis_from_torch_axis
    return tensor

  @staticmethod
  def _base_get_output_shape_from_returnn(inputs_flat: List[Tensor], layer: LayerBase
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

    out_returnn_axis_to_torch_axis = {}
    # Torch would maybe have operated on [B,D_in,T_in] input, and produce [B,D_out,T_out] output.
    naming = Naming.get_instance()
    batch_size = None
    dyn_size_dim_tag_to_spatial_idx_and_torch_dim = OrderedDict()  # RETURNN dim tag -> in spatial idx, Torch dim
    for input in inputs_flat:
      if not isinstance(input, Tensor):
        continue
      if not input.shape:
        continue  # skip scalars
      x = naming.tensors[input]
      assert isinstance(x, TensorEntry)
      assert x.returnn_data and x.returnn_axis_from_torch_axis is not None
      if x.returnn_data.have_batch_axis():
        batch_size = input.shape[x.returnn_axis_from_torch_axis[x.returnn_data.batch_dim_axis]]
      for i in x.returnn_data.get_dynamic_axes():
        dim_tag = x.returnn_data.get_dim_tag(i)
        assert i in x.returnn_data.get_spatial_batch_axes()
        spatial_idx = x.returnn_data.get_spatial_batch_axes().index(i)
        torch_dim = input.shape[x.returnn_axis_from_torch_axis[i]]
        if dim_tag not in dyn_size_dim_tag_to_spatial_idx_and_torch_dim:
          dyn_size_dim_tag_to_spatial_idx_and_torch_dim[dim_tag] = (spatial_idx, torch_dim)

      # Find mapping to layer_output_shape_meta.
      mapping_out_to_in = {}
      for out_axis in range(layer.output.batch_ndim):
        if out_axis == layer.output.batch_dim_axis:
          if x.returnn_data.have_batch_axis():
            mapping_out_to_in[out_axis] = x.returnn_data.batch_dim_axis
          else:
            mapping_out_to_in[out_axis] = None  # new axis
          continue
        if out_axis == layer.output.feature_dim_axis:
          if x.returnn_data.have_feature_axis():
            mapping_out_to_in[out_axis] = x.returnn_data.feature_dim_axis
          else:
            mapping_out_to_in[out_axis] = None  # new axis
          continue
        if out_axis in layer.output.get_dynamic_axes():
          dim_tag = layer.output.get_dim_tag(out_axis)
          if dim_tag in dyn_size_dim_tag_to_spatial_idx_and_torch_dim:
            in_spatial_idx, _ = dyn_size_dim_tag_to_spatial_idx_and_torch_dim[dim_tag]
            mapping_out_to_in[out_axis] = x.returnn_data.get_spatial_batch_axes()[in_spatial_idx]
            continue
        assert out_axis in layer.output.get_spatial_batch_axes()
        out_spatial_idx = layer.output.get_spatial_batch_axes().index(out_axis)
        if len(x.returnn_data.get_spatial_batch_axes()) == len(layer.output.get_spatial_batch_axes()):
          mapping_out_to_in[out_axis] = x.returnn_data.get_spatial_batch_axes()[out_spatial_idx]
          continue
        # Just skip other cases now.

      in_values = [j for (i, j) in sorted(mapping_out_to_in.items()) if j is not None]
      if in_values != sorted(in_values):  # do we have some reordering?
        rem_out = [i for i in range(layer.output.batch_ndim) if i not in mapping_out_to_in]
        rem_in = [i for i in range(x.returnn_data.batch_ndim) if i not in in_values]
        assert len(rem_out) == len(rem_in)  # assumption, otherwise no idea what to do
        assert None not in list(mapping_out_to_in.values())  # no new axis
        mapping_out_to_in.update({i: j for (i, j) in zip(rem_out, rem_in)})

        # Assume no reordering happened on the Torch site.
        for returnn_out_axis, returnn_in_axis in mapping_out_to_in.items():
          if returnn_in_axis is not None:
            out_returnn_axis_to_torch_axis[returnn_out_axis] = x.torch_axis_from_returnn_axis[returnn_in_axis]

      else:  # same order, but maybe some dims added or removed
        if layer.output.batch_ndim == x.returnn_data.batch_ndim:  # no dim added/removed
          for returnn_out_axis, returnn_in_axis in mapping_out_to_in.items():
            if returnn_in_axis is not None:
              out_returnn_axis_to_torch_axis[returnn_out_axis] = x.torch_axis_from_returnn_axis[returnn_in_axis]

        pass  # should be covered below

      break
    assert all(0 <= d < layer.output.batch_ndim for d in out_returnn_axis_to_torch_axis.values())
    assert len(set(out_returnn_axis_to_torch_axis.values())) == len(out_returnn_axis_to_torch_axis)
    # The remaining axes should be the spatial axes.
    rem_torch_axes = set(range(layer.output.batch_ndim)).difference(set(out_returnn_axis_to_torch_axis.values()))
    rem_returnn_axes = set(range(layer.output.batch_ndim)).difference(set(out_returnn_axis_to_torch_axis.keys()))
    assert len(rem_torch_axes) == len(rem_returnn_axes)
    rem_torch_axes_ = sorted(rem_torch_axes)
    rem_returnn_axes_ = sorted(rem_returnn_axes)

    out_shape = list(layer.output.batch_shape)
    if layer.output.have_batch_axis():
      assert batch_size is not None
      out_shape[layer.output.batch_dim_axis] = batch_size
    if layer.output.get_dynamic_axes():
      assert dyn_size_dim_tag_to_spatial_idx_and_torch_dim
      for i in layer.output.get_dynamic_axes():
        dim_tag = layer.output.get_dim_tag(i)
        if dim_tag in dyn_size_dim_tag_to_spatial_idx_and_torch_dim:
          in_spatial_idx, out_shape[i] = dyn_size_dim_tag_to_spatial_idx_and_torch_dim[dim_tag]
          if i in rem_returnn_axes:
            out_spatial_idx = rem_returnn_axes_.index(i)
            if in_spatial_idx != out_spatial_idx:
              out_returnn_axis_to_torch_axis[i] = rem_torch_axes_[in_spatial_idx]
              rem_returnn_axes.remove(i)
              rem_torch_axes.remove(rem_torch_axes_[in_spatial_idx])
        else:
          # Assume same order.
          assert len(layer.output.get_dynamic_axes()) == len(dyn_size_dim_tag_to_spatial_idx_and_torch_dim)
          out_shape[i] = (
            list(dyn_size_dim_tag_to_spatial_idx_and_torch_dim.values())[layer.output.get_dynamic_axes().index(i)][1])
    assert all(d for d in out_shape)

    for i, j in zip(sorted(rem_returnn_axes), sorted(rem_torch_axes)):
      assert i not in out_returnn_axis_to_torch_axis
      out_returnn_axis_to_torch_axis[i] = j
    assert (
        len(set(out_returnn_axis_to_torch_axis.values())) ==
        len(set(out_returnn_axis_to_torch_axis.keys())) ==
        layer.output.batch_ndim)
    torch_axis_to_returnn = {i: j for (j, i) in out_returnn_axis_to_torch_axis.items()}
    assert len(torch_axis_to_returnn) == layer.output.batch_ndim
    out_shape = [out_shape[torch_axis_to_returnn[i]] for i in range(layer.output.batch_ndim)]
    out_returnn_axis_from_torch_axis = torch_axis_to_returnn
    return tuple(out_shape), out_returnn_axis_from_torch_axis

  def _get_output_shape_from_returnn(self,
                                     inputs_flat: List[Tensor], layer: LayerBase
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
    return self._base_get_output_shape_from_returnn(inputs_flat=inputs_flat, layer=layer)

  def as_returnn_torch_functional(self):
    """
    Call this to mark this module such that this would not be used when used with the original Torch.
    Usually this is via the functional API, which calls back to standard Torch modules.

    :return: self
    """
    self.is_original_torch_module = False
    return self

  @classmethod
  def _make_returnn_dummy_input(cls, data: Data) -> numpy.ndarray:
    some_primes = (3, 5, 7, 11, 13)  # use primes for dynamic dims, just nicer to identify in logs
    dynamic_axes = [i for i, dim in enumerate(data.batch_shape) if dim is None]
    assert len(dynamic_axes) <= len(some_primes)  # just not implemented otherwise
    shape = list(data.batch_shape)
    for i, j in enumerate(dynamic_axes):
      shape[j] = some_primes[i]
    return numpy.zeros(shape, dtype=data.dtype)

  def _returnn_dummy_call(self, *returnn_inputs: Dict[str, Any]) -> Naming:
    from pytorch_to_returnn.torch import from_numpy
    naming = Naming.get_instance()
    returnn_datas = []
    for i, kwargs in enumerate(returnn_inputs):
      kwargs = kwargs.copy()
      if "name" not in kwargs:
        kwargs["name"] = "data" if i == 0 else f"data:{i}"
      x = Data(**kwargs)
      returnn_datas.append(x)
    dummy_inputs_np = [self._make_returnn_dummy_input(x) for x in returnn_datas]
    dummy_inputs_torch = [from_numpy(x) for x in dummy_inputs_np]
    for i in range(len(returnn_inputs)):
      naming.register_input(tensor=dummy_inputs_torch[i], returnn_data=returnn_datas[i])
    out = self(*dummy_inputs_torch)
    assert isinstance(out, Tensor)
    naming.register_output(out)
    return naming

  def as_returnn_net_dict(self, *returnn_inputs: Dict[str, Any]):
    return self._returnn_dummy_call(*returnn_inputs).root_namespace.dump_as_returnn_net_dict()

  def as_returnn_layer_dict(self, *returnn_inputs: Dict[str, Any]):
    return self._returnn_dummy_call(*returnn_inputs).root_namespace.dump_as_returnn_layer_dict()
