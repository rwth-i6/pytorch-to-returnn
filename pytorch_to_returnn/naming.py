
from __future__ import annotations

import tensorflow as tf
import numpy
import typing
from typing import Optional, Any, List, Tuple, TypeVar, Dict, Callable, Iterable, Union, Generator
import weakref
from weakref import WeakKeyDictionary, ref
from collections import OrderedDict
from contextlib import contextmanager
import itertools
from returnn.config import Config
from returnn.tf.network import ExternData, TFNetwork
from returnn.tf.layers.basic import InternalLayer, LayerBase, CopyLayer, SubnetworkLayer
from returnn.tf.util.data import Data, DimensionTag

if typing.TYPE_CHECKING:
  # Just for typing. Although we also cover traced/wrapped Torch.
  from .torch import Tensor
  from .torch.nn import Module


class TensorEntry:
  tensor: ref[Tensor]
  returnn_data: Optional[Data] = None
  returnn_axis_to_torch_axis: Optional[Dict[int, int]] = None
  is_param: bool = False
  is_const: bool = False  # e.g. via from_numpy, empty, zeros, etc
  is_input: bool = False  # in TF1 terminology, would be a placeholder
  output_from_modules: List["ModuleEntry"]
  output_from_calls: List["CallEntry"]
  parent_owning_modules: List[Tuple["ModuleEntry", str]]  # e.g. param or buffer
  creation_stack_call: Optional[CallEntry]
  module_context_stack: List[ModuleEntry]
  names: List["RegisteredName"]

  def __init__(self, tensor: ref[Tensor],
               creation_stack_call: Optional[CallEntry], module_context_stack: List[ModuleEntry]):
    self.tensor = tensor
    self.creation_stack_call = creation_stack_call
    self.module_context_stack = module_context_stack
    self.output_from_modules = []
    self.output_from_calls = []
    self.parent_owning_modules = []
    self.names = []

  def __repr__(self):
    if self.returnn_data:
      returnn_data_repr = f"[{','.join(self.returnn_data.get_batch_axes_short_description())}]"
      if self.returnn_axis_to_torch_axis == {i: i for i in range(self.returnn_data.batch_ndim)}:
        mapping_repr = "id"
      else:
        mapping_repr = repr(self.returnn_axis_to_torch_axis).replace(" ", "")
      returnn_data_repr = f"{self.returnn_data.name!r} {returnn_data_repr} axes {mapping_repr}"
    else:
      returnn_data_repr = None
    tensor = self.tensor()
    tensor_repr = repr(tensor.shape).replace(" ", "") if tensor is not None else "?"
    name_repr = self.get_canonical_name(fallback='?')
    if name_repr != "?":
      name_repr = repr(name_repr)
    return (
      f"<{self.__class__.__name__ }"
      f" name:{name_repr}"
      f" tensor:{tensor_repr}"
      f" returnn_data:{returnn_data_repr}"
      f">")

  def get_canonical_parent_module(self, parent_namespace: Optional[RegisteredName] = None) -> Optional[ModuleEntry]:
    if self.parent_owning_modules:
      return self.parent_owning_modules[0][0]
    if self.module_context_stack:
      mod = self.module_context_stack[-1]
      if parent_namespace and mod in parent_namespace.modules:
        pass
      else:
        return mod
    return None

  def get_canonical_name(self, *, fallback: Optional[str] = None) -> str:
    if self.parent_owning_modules:
      return self.parent_owning_modules[0][1]
    if fallback:
      return fallback
    raise NotImplementedError

  def get_returnn_axis_description(self, torch_axis: int) -> str:
    """
    :param torch_axis:
    :return: name such that
      self.returnn_axis_to_torch_axis[self.returnn_data.get_axis_from_description(name)] == torch_axis
    """
    assert self.returnn_data and self.returnn_axis_to_torch_axis is not None
    torch_axis_to_returnn_axis = {i: j for (j, i) in self.returnn_axis_to_torch_axis.items()}
    assert len(torch_axis_to_returnn_axis) == len(self.returnn_axis_to_torch_axis) == self.returnn_data.batch_ndim
    axis = torch_axis_to_returnn_axis[torch_axis]
    if axis == self.returnn_data.batch_dim_axis:
      return "B"
    if axis == self.returnn_data.time_dim_axis:
      return "T"
    if axis == self.returnn_data.feature_dim_axis:
      return "F"
    dim_tag = self.returnn_data.get_dim_tag(axis)
    assert dim_tag.kind == DimensionTag.Types.Spatial
    if dim_tag.dyn_size is not None:
      return f"stag:{dim_tag.description}"
    static_axes = self.returnn_data.get_static_axes()
    if axis in static_axes:
      return f"static:{static_axes.index(axis)}"
    spatial_axes = self.returnn_data.get_spatial_batch_axes()
    if axis in spatial_axes:
      return f"spatial:{spatial_axes.index(axis)}"
    raise Exception(f"Should not get here. Dim tag {dim_tag} for axis {axis} for data {self.returnn_data}")


class CallEntry:
  """
  Can be a module() call, or regular func.
  Note that a module can be called multiple times.
  """
  module: ModuleEntry
  orig_inputs: Optional[Tuple[Union[Tensor, Any]]] = None
  orig_outputs: Optional[Union[Tensor, Tuple[Tensor]]] = None
  inputs: Optional[List["TensorEntry"]] = None
  outputs: Optional[List["TensorEntry"]] = None
  parent_call: Optional["CallEntry"] = None  # parent in the call stack
  child_calls: List["CallEntry"]
  level: Optional[int] = None
  namespace: Optional["RegisteredName"] = None
  returnn_layer: Optional[LayerBase] = None
  returnn_layer_dict: Optional[Dict[str, Any]] = None

  def __init__(self, module: ModuleEntry):
    self.module = module
    module.calls.append(self)
    self.child_calls = []

  def __repr__(self):
    return f"<{self.__class__.__name__} #{self.level} {self.module!r}>"

  def get_root_call(self) -> "CallEntry":
    entry = self
    while entry.parent_call:
      entry = entry.parent_call
    return entry

  def get_canonical_name(self) -> str:
    """
    Considering the canonical context where this is being used.
    Not an absolute name but relative.
    """
    return self.module.get_canonical_name(parent_namespace=self.namespace.parent)

  def set_returnn_layer(self, layer: LayerBase, layer_dict: Optional[Dict[str, Any]]):
    self.returnn_layer = layer
    self.returnn_layer_dict = layer_dict

  def set_outputs(self, outputs: Union[Tensor, Tuple[Tensor], List[Tensor]]):
    assert self.outputs is None
    naming = Naming.get_instance()
    if naming.keep_orig_module_io_tensors:
      self.orig_outputs = outputs
    if naming.wrap_to_returnn_enabled:
      if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
      entry_outputs = [naming.tensors[x] for x in outputs]
      self.outputs = entry_outputs
      for x in entry_outputs:
        x.output_from_calls.append(self)
        if self.module:
          x.output_from_modules.append(self.module)
      if entry_outputs:
        if self.namespace not in entry_outputs[0].names:
          entry_outputs[0].names.append(self.namespace)

  def __enter__(self):
    # Assume via push_func_call
    assert Naming.get_instance().module_call_stack[-1] is self
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    if not exc_type:
      Naming.get_instance().pop_module_call(self)


class _ObjAttr:
  T = TypeVar("T")
  obj: T
  attr: str

  def __init__(self, obj: T, attr: str):
    self.obj = obj
    self.attr = attr


class ModuleEntry:
  module: Module
  level: Optional[int] = None
  calls: List[CallEntry]
  names: List[RegisteredName]
  canonical_name: Optional[str] = None
  parent_owning_modules: List[Tuple["ModuleEntry", str]]
  parent_context_modules: List["ModuleEntry"]

  def __init__(self, module: Module):
    self.module = module
    self.calls = []
    self.names = []
    self.parent_owning_modules = []
    self.parent_context_modules = []

  def __repr__(self):
    module_repr = repr(self.module)
    # torch.nn.Module.__repr__ can be too verbose when there are children...
    # Strip that away.
    if "\n" in module_repr:
      lines = module_repr.splitlines()
      assert len(lines) >= 2
      module_repr = f"{lines[0].strip()}...{lines[-1].strip()}"
    return f"<ModuleEntry {module_repr}>"

  def get_parent_calling_modules(self) -> List["ModuleEntry"]:
    res = []
    for call in self.calls:
      assert call.module is self
      while call.parent_call:
        call = call.parent_call
        if call.module:
          res.append(call.module)
          break
    return res

  def get_root_owning_module(self) -> "ModuleEntry":
    mod = self
    while mod.parent_owning_modules:
      mod = mod.parent_owning_modules[0][0]
    return mod

  def get_canonical_name(self, parent_namespace: Optional[RegisteredName] = None, *, _visited=None) -> str:
    if self.canonical_name:
      return self.canonical_name
    if _visited is None:
      _visited = set()
    _visited.add(self)
    if parent_namespace is None:
      parent_namespace = Naming.get_instance().root_namespace
    if self.parent_owning_modules:
      mod, name = self.parent_owning_modules[0]
      if parent_namespace and mod in parent_namespace.modules:
        prefix = ""
      elif not mod.module.has_torch_forward() and mod not in _visited:
        prefix = mod.get_canonical_name(_visited=_visited) + "_"
      else:
        prefix = ""
      if not prefix and name[:1].isnumeric():
        return f"layer{name}"
      return prefix + name
    if parent_namespace and self in parent_namespace.modules:
      return self.module.get_returnn_name()
    if set(self.parent_context_modules).intersection(_visited):
      return self.module.get_returnn_name()
    prefix = ""
    for mod in reversed(self.parent_context_modules):
      prefix = mod.get_canonical_name(_visited=_visited, parent_namespace=parent_namespace) + "_"
      break
    return prefix + self.module.get_returnn_name()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    if not exc_type:
      Naming.get_instance().pop_module_context(self.module)


_SearchType = TypeVar("_SearchType")
_SearchChildsFuncT = Callable[[_SearchType], Iterable[_SearchType]]
_SearchFilterFuncT = Callable[[_SearchType], bool]


def _breadth_first_search(
      root: _SearchType = None, *,
      seed: List[_SearchType] = None,
      childs: _SearchChildsFuncT, filter_elem: Optional[_SearchFilterFuncT] = None):
  visited = set()
  if seed:
    assert root is None
    queue = seed
  else:
    queue = [root]
  while queue:
    next_queue = []
    for elem in queue:
      if elem in visited:
        continue
      visited.add(elem)
      if filter_elem and not filter_elem(elem):
        continue
      yield elem
      next_queue.extend(childs(elem))
    queue = next_queue


def _breadth_first_search_first(
      root: _SearchType, childs: _SearchChildsFuncT, filter_elem: Optional[_SearchFilterFuncT] = None):
  for elem in _breadth_first_search(root=root, childs=childs, filter_elem=filter_elem):
    return elem
  return None


class RegisteredName:
  childs_by_name: OrderedDict[str, "RegisteredName"]
  parent: Optional["RegisteredName"]
  name: Optional[str]  # if parent
  level: int = 0
  modules: List[ModuleEntry]  # can be multiple merged together
  calls: List[CallEntry]  # can be multiple merged together. can be empty if this is some input
  tensor: Optional[TensorEntry] = None  # output from the call
  returnn_ctx: Optional[ReturnnContext] = None

  def __init__(self, *,
               wrap_to_returnn_enabled: Optional[bool] = None,
               parent: Optional["RegisteredName"] = None, name: Optional[str] = None,
               call: Optional[CallEntry] = None,
               tensor: Optional[TensorEntry] = None):
    self.childs_by_name = OrderedDict()
    self.parent = parent
    if parent:
      assert name
      assert wrap_to_returnn_enabled is None
      wrap_to_returnn_enabled = parent.wrap_to_returnn_enabled
    else:
      assert not name
      assert wrap_to_returnn_enabled is not None
    self.name = name
    self.wrap_to_returnn_enabled = wrap_to_returnn_enabled
    self.modules = []
    self.calls = []
    if call:
      self.assign_call(call)
    if parent:
      self.level = parent.level + 1
    if tensor:
      self.assign_tensor(tensor)
    if self.wrap_to_returnn_enabled:
      if not parent:  # with parent, returnn_ctx will be created once needed
        self._create_returnn_ctx()

  def __repr__(self):
    return f"<{self.__class__.__name__} {self.get_absolute_name()!r}>"

  def get_absolute_name(self):
    names = []
    name_ = self
    while name_.parent:
      names.insert(0, name_.name)
      name_ = name_.parent
    return "/".join(names) if names else ""

  def assign_tensor(self, tensor: TensorEntry):
    assert not self.tensor
    self.tensor = tensor
    tensor.names.append(self)

  def assign_call(self, call: CallEntry):
    if call.module:
      self.assign_module(call.module)
    assert all(not c.module or c.module.module.has_torch_forward() for c in self.calls)
    if self.calls:
      assert not call.module or call.module.module.has_torch_forward()
    assert not call.namespace
    call.namespace = self
    self.calls.append(call)

  def _create_returnn_ctx(self):
    if self.parent:
      if not self.parent.returnn_ctx:
        self.parent._create_returnn_ctx()
      assert self.parent.returnn_ctx
    self.returnn_ctx = ReturnnContext(
      parent=self.parent.returnn_ctx if self.parent else None,
      name=self.name)

  def assign_module(self, module: ModuleEntry):
    if module in self.modules:
      return
    self.modules.append(module)
    module.names.append(self)
    if self.wrap_to_returnn_enabled:
      if module.module.has_torch_forward() and not self.returnn_ctx:
        # Need our own returnn ctx / subnet.
        assert self.parent
        self._create_returnn_ctx()
      if self.returnn_ctx:
        assert all(m.module.has_torch_forward() for m in self.modules)

  def _get_unique_name(self, suggested_name: str) -> str:
    if suggested_name not in self.childs_by_name:
      return suggested_name
    for i in itertools.count(1):
      suggested_name_ = f"{suggested_name}_{i}"
      if suggested_name_ not in self.childs_by_name:
        return suggested_name_

  def register(self, *, suggested_name: str, call: Optional[CallEntry] = None) -> RegisteredName:
    name = self._get_unique_name(suggested_name)
    name_ = RegisteredName(parent=self, name=name, call=call)
    self.childs_by_name[name] = name_
    return name_

  def register_input(self, name: str, tensor: TensorEntry) -> RegisteredName:
    if name in self.childs_by_name:
      assert self.childs_by_name[name].tensor is tensor
      return self.childs_by_name[name]
    assert name not in self.childs_by_name
    name_ = RegisteredName(parent=self, name=name, tensor=tensor)
    self.childs_by_name[name] = name_
    if self.wrap_to_returnn_enabled:
      self.returnn_ctx.define_input(tensor)
    return name_

  def name_for_tensor(self, tensor: TensorEntry) -> str:
    for name_ in tensor.names:
      if name_.parent is self:
        assert self.childs_by_name[name_.name] is name_
        return name_.name
    # If you get here, check the logic in Module.__call__, Naming.push_module_call.
    raise KeyError(f"namespace {self!r}: tensor {tensor!r} not found")

  def find_name_for_module(self, module: ModuleEntry) -> Optional[str]:
    for name, child in self.childs_by_name.items():
      if module in child.modules:
        return name
    return None

  def dump(self, prefix=""):
    for name, child in self.childs_by_name.items():
      if name.startswith("."):
        print(f"{prefix}{name}: (hidden)")
        continue
      if len(child.modules) == 0:
        mod = None
      elif len(child.modules) == 1:
        mod = child.modules[0]
      else:
        mod = child.modules
      if len(child.calls) == 0:
        res = None
      elif len(child.calls) == 1:
        res = child.calls[0].outputs
        if res is None:
          res = "..."
        elif len(res) == 0:
          res = f"<{child.calls[0]} without outputs>"
        elif len(res) == 1:
          res = res[0]
      else:
        res = f"<multiple calls {child.calls}>"
      print(f"{prefix}{name}: {mod} -> {res}")
      child.dump(prefix=f"{prefix}  ")

  def dump_as_returnn_layer_dict(self):
    if self.calls and not self.calls[0].module.module.has_torch_forward():
      assert len(self.calls) == 1
      call = self.calls[0]
      return call.returnn_layer_dict
    # Subnetwork
    input_child = self.childs_by_name["data"]
    input_tensor = input_child.tensor
    assert input_tensor
    assert self.parent
    parent_namespace = self.parent
    input_layer_name = parent_namespace.name_for_tensor(input_tensor)
    subnet_dict = self.dump_as_returnn_net_dict()
    return {"class": "subnetwork", "from": input_layer_name, "subnetwork": subnet_dict}

  def dump_as_returnn_net_dict(self) -> Dict[str, Dict[str, Any]]:
    net_dict = {}
    for name, child in self.childs_by_name.items():
      if not child.calls:
        continue  # e.g. input "data"
      net_dict[name] = child.dump_as_returnn_layer_dict()
    return net_dict


class ReturnnContext:
  def __init__(self, *, parent: Optional[ReturnnContext] = None, name: Optional[str] = None):
    self.parent = parent
    if parent:
      assert name
      self.config = parent.config
      self.tf_name_scope = parent.network.get_absolute_name_scope_prefix() + LayerBase.cls_get_tf_scope_name(name)
      assert parent.network.extern_data.data
      self._sub_layer = (
        parent.network.add_layer(
          name=name, layer_class=SubnetworkLayer,
          # This is just a placeholder, will be replaced in define_output.
          sources=[parent.network.get_layer("data")],
          subnetwork={"output": {"class": "copy"}}))  # type: SubnetworkLayer
      self._dummy_sub_output = self._sub_layer.subnetwork.layers["output"]
    else:
      self.config = Config({
        # "debug_print_layer_output_template": True,
      })
      self.tf_name_scope = ""
      self._sub_layer = None
      self._dummy_sub_output = None
    if self._sub_layer:
      self.network = self._sub_layer.subnetwork
    else:
      assert not parent
      self.network = TFNetwork(
        extern_data=ExternData(), config=self.config, name="root",
        absolute_name_prefix=(self.tf_name_scope + "/") if self.tf_name_scope else "")

  def __repr__(self):
    return f"<{self.__class__.__name__} {self.network.get_absolute_name_prefix()!r}>"

  def define_input(self, input: TensorEntry):
    if self._dummy_sub_output:
      assert self.network.layers["output"] is self._dummy_sub_output
      self._dummy_sub_output = None
      # Reset both, as we refill them. They contain dummy data.
      self.network.layers.clear()
      self.network.extern_data.data.clear()
    # TODO hardcoded defaults
    data_key = "data"
    assert data_key not in self.network.extern_data.data
    assert input.returnn_data
    self.network.extern_data.data[data_key] = input.returnn_data

  def define_output(self, layer_name: str) -> LayerBase:
    assert layer_name in self.network.layers
    if "output" in self.network.layers:
      del self.network.layers["output"]  # just redefine...  # TODO better?
    self.network.construct_layer({"output": {"class": "copy", "from": layer_name}}, name="output")
    layer = self.network.layers["output"]
    if self._sub_layer:
      self._sub_layer.output = layer.output
      return self._sub_layer
    return layer


class Naming:
  tensors: WeakKeyDictionary[Tensor, TensorEntry]
  const_tensor_cache: List[Tensor]
  modules: OrderedDict[Module, ModuleEntry]
  inputs: List[Tensor]
  outputs: List[Tensor]
  module_creation_stack: List[ModuleEntry]
  module_apply_stack: List[ModuleEntry]
  module_context_stack: List[ModuleEntry]
  module_call_stack: List[CallEntry]
  root_func_calls: List[CallEntry]
  _instance: Optional[Naming] = None

  @classmethod
  @contextmanager
  def make_instance(cls, **kwargs) -> Generator[Naming]:
    assert not cls._instance
    ctx = Naming(**kwargs)
    cls._instance = ctx
    yield ctx
    assert cls._instance is ctx
    cls._instance = None

  @classmethod
  def get_instance(cls) -> Naming:
    assert cls._instance
    return cls._instance

  def __init__(self, *,
               wrap_to_returnn_enabled: bool,
               keep_orig_module_io_tensors: bool,
               import_params_from_torch_namespace: Optional[Naming] = None
               ):
    """
    :param wrap_to_returnn_enabled: Will construct corresponding RETURNN layers.
    :param keep_orig_module_io_tensors: Keeps references to the original (or wrapped) torch.Tensor instances
       for all module calls. This will need extra memory.
    :param import_params_from_torch_namespace: If given, we try to import params.
    """
    self.wrap_to_returnn_enabled = wrap_to_returnn_enabled
    self.keep_orig_module_io_tensors = keep_orig_module_io_tensors
    self.import_params_from_torch_namespace = import_params_from_torch_namespace
    self.tensors = WeakKeyDictionary()
    self.const_tensor_cache = []
    self.modules = OrderedDict()
    self.inputs = []
    self.outputs = []
    self.module_context_stack = []
    self.module_creation_stack = []
    self.module_apply_stack = []
    self.module_call_stack = []
    self.root_func_calls = []
    self.root_namespace = RegisteredName(parent=None, wrap_to_returnn_enabled=wrap_to_returnn_enabled)
    self.tmp_eager_root_namespace = self.root_namespace.register(suggested_name=".tmp_root")

  @contextmanager
  def push_module_creation(self, module: Module) -> ModuleEntry:
    assert module not in self.modules
    entry = ModuleEntry(module=module)
    self.modules[module] = entry
    self.module_creation_stack.append(entry)
    with self.push_module_context(module):
      yield entry
    assert self.module_creation_stack[-1] is entry
    self.module_creation_stack.pop(-1)

  @contextmanager
  def push_module_apply(self, module: Module) -> ModuleEntry:
    entry = self.modules[module]
    self.module_apply_stack.append(entry)
    with self.push_module_context(module):
      yield entry
    assert self.module_apply_stack[-1] is entry
    self.module_apply_stack.pop(-1)

  def push_module_context(self, module: Module) -> ModuleEntry:
    entry = self.modules[module]
    if self.module_context_stack:
      prev_top = self.module_context_stack[-1]
      if prev_top not in entry.parent_context_modules and prev_top.module is not module:
        entry.parent_context_modules.append(prev_top)
    self.module_context_stack.append(entry)
    return entry

  def pop_module_context(self, module: Module):
    entry = self.modules[module]
    assert self.module_context_stack[-1] is entry
    self.module_context_stack.pop(-1)

  def _prepare_module_call_returnn_inputs(self, call: CallEntry):
    """
    It means this module has no forward, i.e. it is wrapped as RETURNN layer.
    We might need to make some inputs available, which are not available yet,
    e.g. constants, params, etc.
    """
    if not self.wrap_to_returnn_enabled:
      return
    for x in call.inputs:
      if x is None:
        continue
      assert isinstance(x, TensorEntry)
      if not x.names:
        if x.is_param:
          assert x.returnn_data
          assert x.returnn_data.placeholder is None
          from .torch.nn.modules import Variable
          param_name = x.get_canonical_name()
          if x.returnn_data.name == "_unnamed_param":
            x.returnn_data.name = f"param:{param_name}"
          parent_mod = x.get_canonical_parent_module()
          prefix = (parent_mod.get_canonical_name() + "_") if parent_mod else ""
          mod = Variable(param=x.tensor())
          self.modules[mod].canonical_name = prefix + param_name
          res = mod()
          res_tensor = self.tensors[res]
          assert isinstance(res_tensor, TensorEntry)
          assert len(res_tensor.names) == 1
          assert res_tensor.returnn_data.placeholder is not None
          x.returnn_data.placeholder = res_tensor.returnn_data.placeholder
        elif not x.output_from_calls or x.is_const:
          # Assume this is a constant.
          const_name = x.get_canonical_name(fallback="unnamed_const")
          tensor = x.tensor()
          if not x.returnn_data:
            x.returnn_data = Data(
              name=f"const:{const_name}", shape=tensor.shape, dtype=tensor.dtype.name,
              batch_dim_axis=None, time_dim_axis=None)
            x.returnn_axis_to_torch_axis = {i: i for i in range(len(tensor.shape))}
          parent_mod = x.get_canonical_parent_module()
          prefix = (parent_mod.get_canonical_name() + "_") if parent_mod else ""
          from .torch.nn.modules import Constant
          mod = Constant(value=tensor)
          self.modules[mod].canonical_name = prefix + const_name
          res = mod()
          res_tensor = self.tensors[res]
          assert isinstance(res_tensor, TensorEntry)
          assert len(res_tensor.names) == 1
          assert res_tensor.returnn_data.placeholder is not None
          x.returnn_data.placeholder = res_tensor.returnn_data.placeholder
          x.is_const = True
        else:
          raise Exception(f"Cannot handle tensor {x}, via {x.output_from_calls} ...")

  def push_module_call(self, *, module: Module, inputs: List[Tensor]) -> CallEntry:
    module_entry = self.modules[module]
    entry = CallEntry(module=module_entry)
    if self.keep_orig_module_io_tensors:
      entry.orig_inputs = inputs
    entry.inputs = [self._make_tensor(x) for x in inputs]
    entry.level = len(self.module_call_stack)
    if self.module_call_stack:
      recent_entry = self.module_call_stack[-1]
      recent_entry.child_calls.append(entry)
      entry.parent_call = recent_entry
    else:
      self.root_func_calls.append(entry)
    if self.module_creation_stack or self.module_apply_stack:
      # This is likely for parameter initialization, or setting up constants.
      # Keep this in a separate namespace.
      root_namespace = self.tmp_eager_root_namespace
    else:
      root_namespace = self.root_namespace
    if entry.parent_call:
      assert entry.parent_call.namespace
      parent_namespace = entry.parent_call.namespace
      if self.wrap_to_returnn_enabled:
        while not parent_namespace.returnn_ctx:  # e.g. if call within another module non-forward call
          assert parent_namespace.parent
          parent_namespace = parent_namespace.parent
    else:
      parent_namespace = root_namespace
    # Find right parent namespace.
    parents_hierarchy = []
    parent_module = module_entry
    while parent_module not in parent_namespace.modules and parent_module not in parents_hierarchy:
      parents_hierarchy.append(parent_module)
      if not parent_module.parent_owning_modules:
        if parent_module not in self.module_context_stack and self.module_context_stack:
          parent_module = self.module_context_stack[-1]
          continue  # try further
        elif parent_module in self.module_context_stack and self.module_context_stack.index(parent_module) > 0:
          parent_module = self.module_context_stack[self.module_context_stack.index(parent_module) - 1]
          continue  # try further
        # no parent anymore, so use root namespace in any case
        parent_namespace = root_namespace
        break
      parent_module = parent_module.parent_owning_modules[0][0]  # could do search over all, but just use first for now
    for parent_module in reversed(parents_hierarchy):
      assert parent_module not in parent_namespace.modules
      if parent_module is not module_entry and not parent_module.module.has_torch_forward():
        # Skip.
        parent_module = module_entry
      if parent_namespace is root_namespace and _flatten_namespace_for_mod(parent_module):
        # Special case, somewhat nicer to flatten the namespace for this case.
        root_namespace.assign_module(parent_module)
      else:
        name = parent_namespace.find_name_for_module(parent_module)
        if name:
          parent_namespace = parent_namespace.childs_by_name[name]
        else:
          parent_namespace = parent_namespace.register(
            suggested_name=parent_module.get_canonical_name(parent_namespace=parent_namespace))
          parent_namespace.assign_module(parent_module)
      assert parent_module in parent_namespace.modules
      if parent_module is module_entry:
        break
    assert parent_module in parent_namespace.modules and parent_module is module_entry
    namespace = parent_namespace
    assert module_entry in namespace.modules
    namespace.assign_call(entry)
    self.module_call_stack.append(entry)
    assert entry.namespace
    self._prepare_module_call_returnn_inputs(entry)
    return entry

  def pop_module_call(self, call: CallEntry):
    assert self.module_call_stack[-1] is call
    self.module_call_stack.pop(-1)

  def register_module_child_attr(self, parent: Module, attr: str, child: Union[Module, Tensor]):
    assert getattr(parent, attr) is child
    parent_entry = self.modules[parent]
    if child in self.modules:
      child_entry = self.modules[child]
      assert isinstance(child_entry, ModuleEntry)
    else:
      assert child in self.tensors
      child_entry = self.tensors[child]
      assert isinstance(child_entry, TensorEntry)
      for parent_param in parent.parameters(recurse=False):
        if parent_param is child:
          child_entry.is_param = True
          break
    if (parent_entry, attr) not in child_entry.parent_owning_modules:
      child_entry.parent_owning_modules.append((parent_entry, attr))

  def register_tensor(self, tensor: Tensor) -> TensorEntry:
    if tensor not in self.tensors:
      self.tensors[tensor] = TensorEntry(
        tensor=ref(tensor),
        creation_stack_call=self.module_call_stack[-1] if self.module_call_stack else None,
        module_context_stack=list(self.module_context_stack))
    return self.tensors[tensor]

  def _make_tensor(self, x: Union[Tensor, int, float, numpy.number, numpy.ndarray]) -> Optional[TensorEntry]:
    if x is None:
      return None
    if not self.wrap_to_returnn_enabled:
      if x in self.tensors:
        return self.tensors[x]
      return None
    from .torch import from_numpy, Tensor
    if isinstance(x, (int, float, numpy.number, numpy.ndarray)):
      x = from_numpy(x)
      self.const_tensor_cache.append(x)
    assert isinstance(x, Tensor)
    return self.tensors[x]

  def register_input(self, tensor: Tensor, returnn_data: Data) -> Data:
    entry = self.register_tensor(tensor)
    entry.is_input = True
    self.inputs.append(tensor)
    assert tensor.dim() == returnn_data.batch_ndim
    assert all([dim in {tensor.shape[i], None} for i, dim in enumerate(returnn_data.batch_shape)])
    entry.returnn_data = Data(
      name=returnn_data.name, auto_create_placeholders=True,
      dim=returnn_data.dim,
      shape=returnn_data.shape,
      batch_dim_axis=returnn_data.batch_dim_axis,
      time_dim_axis=returnn_data.time_dim_axis,
      feature_dim_axis=returnn_data.feature_dim_axis_or_unspecified,
      available_for_inference=True)
    entry.returnn_axis_to_torch_axis = {i: i for i in range(returnn_data.batch_ndim)}
    # "data" is a special layer name in RETURNN, representing input data
    self.root_namespace.register_input(name=returnn_data.name, tensor=entry)
    assert entry.returnn_data
    return entry.returnn_data

  @staticmethod
  def _register_call_names(root: RegisteredName, calls: List[CallEntry]):
    for call in calls:
      child = root.register(suggested_name=call.get_canonical_name(), call=call)
      Naming._register_call_names(child, call.child_calls)

  def register_output(self, tensor: Tensor) -> Tuple[Data, Dict[int, int]]:
    assert tensor in self.tensors
    entry = self.tensors[tensor]
    assert isinstance(entry, TensorEntry)
    assert not entry.is_param and not entry.is_const and not entry.is_input  # not implemented, although simple...
    self.outputs.append(tensor)
    name = self.root_namespace.name_for_tensor(entry)
    self.root_namespace.returnn_ctx.define_output(name)
    return entry.returnn_data, entry.returnn_axis_to_torch_axis

  def get_module_abs_name(self, module: Module) -> str:
    parts = []
    mod_entry = self.modules[module]
    while True:
      if mod_entry in self.root_namespace.modules:
        break
      name = self.root_namespace.find_name_for_module(mod_entry)
      if name:
        parts.append(name)
        break
      if mod_entry.parent_owning_modules:
        mod_entry, name = mod_entry.parent_owning_modules[0]
        parts.append(name)
        continue
      raise Exception(f"no name for mod {module}")
    abs_name = ".".join(reversed(parts))
    checked_mod = self.get_module_by_abs_name(abs_name)
    assert checked_mod is module
    return abs_name

  def get_module_by_abs_name(self, name: str) -> Module:
    namespace = self.root_namespace
    if name:
      for part_name in name.split("."):
        if part_name[:1].isnumeric():
          part_name = f"layer{part_name}"
        namespace = namespace.childs_by_name[part_name]
    assert namespace.modules
    return namespace.modules[0].module

  def get_module_call_idx(self, *, module: Module, call: CallEntry) -> int:
    mod_entry = self.modules[module]
    assert call in mod_entry.calls
    return mod_entry.calls.index(call)

  def get_module_calls(self, module: Module) -> List[CallEntry]:
    return self.modules[module].calls

  def get_root_module_calls(self) -> OrderedDict[str, CallEntry]:
    d = OrderedDict()
    for name, sub in self.root_namespace.childs_by_name.items():
      if sub.calls:
        d[name] = sub.calls[0]
    return d

  def get_modules_with_params_by_abs_name(self) -> OrderedDict[str, Module]:
    d = OrderedDict()
    _visited = set()

    def visit(namespace: RegisteredName, prefix: str):
      for mod in namespace.modules:
        if mod.module in _visited:
          return
        if not list(mod.module.parameters(recurse=False)):
          continue
        assert prefix and prefix.endswith(".") and len(namespace.modules) == 1
        d[prefix[:-1]] = mod.module
        _visited.add(mod.module)
      for name, sub in namespace.childs_by_name.items():
        if name.startswith("."):
          continue  # skip hidden sub spaces
        visit(namespace=sub, prefix=f"{prefix}{name}.")

    visit(namespace=self.root_namespace, prefix="")
    return d

  def get_returnn_layer_from_module(self, module: Module) -> LayerBase:
    assert self.wrap_to_returnn_enabled
    entry = self.modules[module]
    assert entry.calls
    call = entry.calls[0]
    assert call.returnn_layer
    return call.returnn_layer


def _flatten_namespace_for_mod(mod_entry: ModuleEntry) -> bool:
  if mod_entry.parent_owning_modules:
    # Use the parent module.
    return False
  mod = mod_entry.module
  if not list(mod.children()):
    # For RETURNN wrapped modules, e.g. it means that it has no forward, but wraps to a RETURNN layer.
    # But more generally, if there are no children modules, we assume this directly does some operation.
    # So, keep this as a separate item, do not flatten it.
    return False
  return True
