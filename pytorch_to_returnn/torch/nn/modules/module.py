

from collections import OrderedDict
from typing import Optional, Callable, TypeVar, Iterator, Tuple, Union, Dict, Any, overload
import itertools
from ..parameter import Parameter
from ...tensor import Tensor
from ...autograd import no_grad
from ...utils.hooks import RemovableHandle
from ....naming import Naming


# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar('T', bound='Module')


class Module:
  """
  Base class.
  """
  def __new__(cls, *args, **kwargs):
    res = super(Module, cls).__new__(cls)
    assert isinstance(res, Module)
    Naming.get_instance().push_module_creation(res)
    res.__init__(*args, **kwargs)
    Naming.get_instance().pop_module_creation(res)
    return res

  def __init__(self):
    self._parameters = OrderedDict()  # type: OrderedDict[str, Parameter]
    self._modules = OrderedDict()  # type: OrderedDict[str, Optional[Module]]
    self._buffers = OrderedDict()  # type: OrderedDict[str, Tensor]
    self._non_persistent_buffers_set = set()
    self._forward_pre_hooks = OrderedDict()

  def __repr__(self):
    return f"<{self.__class__.__name__}>"

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
          input_param = Tensor(*input_param.shape)
          # TODO copy...

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

  def __call__(self, *input, **kwargs):
    assert not kwargs  # not implemented yet
    call_entry = Naming.get_instance().push_func_call(module=self, func=self, inputs=list(input))
    for hook in self._forward_pre_hooks.values():
      result = hook(self, input)
      if result is not None:
        if not isinstance(result, tuple):
          result = (result,)
        input = result
    if self.forward:
      assert not self.create_returnn_layer_dict
      assert len(input) == 1  # TODO ...
      call_entry.namespace.register_input(name="data", tensor=Naming.get_instance().tensors[input[0]])
      res = self.forward(*input, **kwargs)
    else:
      assert self.create_returnn_layer_dict
      assert len(input) == 1  # TODO...
      res = Tensor(input[0])  # TODO...
    assert isinstance(res, Tensor)
    Naming.get_instance().pop_func_call(func=self, outputs=[res])
    return res

  # Define either or, but not both.
  forward: Optional[Callable] = None
  create_returnn_layer_dict: Optional[Callable[[str], Dict[str, Any]]] = None
