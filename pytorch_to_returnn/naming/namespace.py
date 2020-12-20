
from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
from collections import OrderedDict
import itertools
from . import call as _call
from . import module as _module
from . import tensor as _tensor
from . import returnn_ctx as _returnn_ctx
from . import naming as _naming


class RegisteredName:
  childs_by_name: OrderedDict[str, RegisteredName]
  parent: Optional[RegisteredName]
  name: Optional[str]  # if parent
  level: int = 0
  modules: List[_module.ModuleEntry]  # can be multiple merged together
  calls: List[_call.CallEntry]  # can be multiple merged together. can be empty if this is some input
  tensor: Optional[_tensor.TensorEntry] = None  # output from the call
  returnn_ctx: Optional[_returnn_ctx.ReturnnContext] = None
  is_reserved: bool  # e.g. input "data" or output "output"
  is_subnet: bool
  is_alternative_subnet_output: bool = False
  ReservedInputName = "data"
  ReservedOutputName = "output"
  ReservedNames = {ReservedInputName, ReservedOutputName}
  _inputs: List[RegisteredName]

  def __init__(self, *,
               wrap_to_returnn_enabled: Optional[bool] = None,
               returnn_train_flag: Optional[bool] = None,
               parent: Optional[RegisteredName] = None, name: Optional[str] = None,
               call: Optional[_call.CallEntry] = None,
               tensor: Optional[_tensor.TensorEntry] = None,
               is_reserved: bool = False, is_subnet: bool):
    self.childs_by_name = OrderedDict()
    self._inputs = []
    self.parent = parent
    if parent:
      assert name
      assert wrap_to_returnn_enabled is None
      assert returnn_train_flag is None
      wrap_to_returnn_enabled = parent.wrap_to_returnn_enabled
      returnn_train_flag = parent.returnn_train_flag
    else:
      assert not name
      assert wrap_to_returnn_enabled is not None
      assert returnn_train_flag is not None
    self.name = name
    self.wrap_to_returnn_enabled = wrap_to_returnn_enabled
    self.returnn_train_flag = returnn_train_flag
    self.is_reserved = is_reserved
    if not is_reserved:
      assert name not in self.ReservedNames
      # If reserved, we also allow other names...
    self.is_subnet = is_subnet
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
        self.maybe_create_returnn_ctx()

  def __repr__(self):
    return f"<{self.__class__.__name__} {self.get_absolute_name()!r} {self._repr_content()}>"

  def _repr_content(self) -> str:
    if len(self.modules) == 0:
      mod = None
    elif len(self.modules) == 1:
      mod = self.modules[0]
    else:
      mod = self.modules
    if len(self.calls) == 0:
      res = None
    elif len(self.calls) == 1:
      res = self.calls[0].outputs_flat
      if res is None:
        res = "..."
      elif len(res) == 0:
        res = f"<{self.calls[0]} without outputs>"
      elif len(res) == 1:
        res = res[0]
    else:
      res = f"<multiple calls {self.calls}>"
    return f"{mod} -> {res}"

  def get_absolute_name(self):
    names = []
    name_ = self
    while name_.parent:
      names.insert(0, name_.name)
      name_ = name_.parent
    return "/".join(names) if names else ""

  def get_parents_hierarchy(self) -> List[RegisteredName]:
    res = []
    x = self
    while x.parent:
      if x.name.startswith("."):  # stop at hidden
        break
      res.append(x.parent)
      x = x.parent
    return res

  def assign_tensor(self, tensor: _tensor.TensorEntry):
    if self.tensor:
      self.tensor.names.remove(self)
    self.tensor = tensor
    tensor.names.append(self)

  def assign_call(self, call: _call.CallEntry):
    if call in self.calls:
      return
    self.assign_module(call.module)
    if self.is_subnet:
      assert call.module.module.has_torch_forward()
    else:
      assert not self.calls  # cannot have multiple calls assigned
      assert not call.module.module.has_torch_forward()
    assert not call.namespace
    call.namespace = self
    self.calls.append(call)

  def maybe_create_returnn_ctx(self):
    """
    Makes sure that returnn_ctx is created.
    """
    assert self.is_subnetwork()
    assert self.wrap_to_returnn_enabled
    if self.returnn_ctx:
      return
    if self.parent:
      if not self.parent.returnn_ctx:
        self.parent.maybe_create_returnn_ctx()
      assert self.parent.returnn_ctx
    self.returnn_ctx = _returnn_ctx.ReturnnContext(
      parent=self.parent.returnn_ctx if self.parent else None,
      name=self.name, returnn_train_flag=self.returnn_train_flag)

  def assign_module(self, module: _module.ModuleEntry):
    if module in self.modules:
      return
    if self.is_subnet:
      assert module.module.has_torch_forward()
    else:
      assert not self.modules  # cannot have multiple modules assigned
      assert not module.module.has_torch_forward()
    self.modules.append(module)
    module.names.append(self)
    if self.is_reserved:
      assert not module.module.has_torch_forward()
    if self.wrap_to_returnn_enabled:
      if module.module.has_torch_forward() and not self.returnn_ctx:
        # Need our own returnn ctx / subnet.
        assert self.parent
        self.maybe_create_returnn_ctx()

  def is_subnetwork(self) -> bool:
    """
    If True, this would be wrapped as a subnetwork in RETURNN.
    If False, this directly maps to a layer in RETURNN.
    """
    return self.is_subnet

  def _get_unique_name(self, suggested_name: str) -> str:
    if suggested_name not in self.childs_by_name and suggested_name not in self.ReservedNames:
      return suggested_name
    for i in itertools.count(1):
      suggested_name_ = f"{suggested_name}_{i}"
      if suggested_name_ not in self.childs_by_name and suggested_name_ not in self.ReservedNames:
        return suggested_name_

  def register_sub_net(self, *, suggested_name: str) -> RegisteredName:
    assert self.is_subnetwork()
    name = self._get_unique_name(suggested_name)
    child = RegisteredName(parent=self, name=name, is_subnet=True)
    self.childs_by_name[name] = child
    return child

  def register_sub_call(self, call: _call.CallEntry) -> RegisteredName:
    assert self.is_subnetwork()
    name = self._get_unique_name(call.module.get_canonical_name(parent_namespace=self))
    child = RegisteredName(parent=self, name=name, is_subnet=call.module.module.has_torch_forward())
    self.childs_by_name[name] = child
    child.assign_call(call)
    return child

  def register_input(self, tensor: _tensor.TensorEntry) -> RegisteredName:
    assert self.is_subnetwork()
    name = self.ReservedInputName
    assert tensor not in self._inputs
    idx = len(self._inputs)
    if idx != 0:
      name += f":{idx}"  # should be consistent with RETURNN SubnetworkLayer concat_sources=False input naming logic
    assert name not in self.childs_by_name
    name_ = RegisteredName(parent=self, name=name, tensor=tensor, is_reserved=True, is_subnet=False)
    self.childs_by_name[name] = name_
    self._inputs.append(name_)
    if self.wrap_to_returnn_enabled:
      self.returnn_ctx.define_input(tensor, data_key=str(idx) if idx else None)
    return name_

  def register_returnn_subnet_output(self, tensor: _tensor.TensorEntry) -> RegisteredName:
    assert self.is_subnetwork()
    from pytorch_to_returnn.torch.nn import Copy
    naming = _naming.Naming.get_instance()
    assert naming.wrap_to_returnn_enabled
    _, reg_name = self.name_for_tensor(tensor)
    potential_calls = set(tensor.output_from_calls).intersection(set(reg_name.calls))
    assert len(potential_calls) == 1, f"{tensor.output_from_calls} vs {reg_name.calls}"
    call = list(potential_calls)[0]
    self.assign_tensor(tensor)  # for this subnet
    name = self.ReservedOutputName
    assert name not in self.childs_by_name
    child = RegisteredName(parent=self, name=name, is_reserved=True, is_subnet=False)
    self.childs_by_name[name] = child
    copy_mod = Copy()
    copy_call = _call.CallEntry(module=naming.modules[copy_mod])
    copy_call.parent_call = call
    call.child_calls.append(copy_call)
    copy_call.inputs_args = (tensor,)
    copy_call.inputs_flat = [tensor]
    copy_call.inputs_kwargs = {}
    child.assign_call(copy_call)
    naming.module_call_stack.append(copy_call)
    try:
      copy_call.apply_call()
    finally:
      assert copy_call is naming.module_call_stack[-1]
      naming.module_call_stack.pop(-1)
    self.returnn_ctx.define_output(copy_call.returnn_layer)
    tensor.returnn_data = copy_call.returnn_layer.output
    return child

  def name_in_ctx(self, possible_sub_names: List[RegisteredName],
                  _src_tensor: Optional[_tensor.TensorEntry] = None) -> Tuple[str, RegisteredName]:
    assert self.is_subnetwork()
    # First try to find in our direct namespace.
    # Only if that fails, search in potential sub namespaces.
    parent_idx = 0
    while True:
      parent_indices = []
      for name_ in possible_sub_names:
        parent_hierarchy = name_.get_parents_hierarchy()
        if self not in parent_hierarchy:
          continue
        if parent_hierarchy[parent_idx] is self:
          if parent_idx == 0:
            assert self.childs_by_name[name_.name] is name_
            return name_.name, name_
          name_.is_alternative_subnet_output = True
          return "/".join([parent.name for parent in reversed(parent_hierarchy[:parent_idx])] + [name_.name]), name_
        parent_indices.append(parent_hierarchy.index(self))
      if not parent_indices:
        # If you get here, check the logic in Module.__call__, Naming.push_module_call.
        raise KeyError(f"namespace {self!r}: {_src_tensor or possible_sub_names!r} not found")
      assert parent_idx == 0  # should not get here twice
      parent_idx = min(parent_indices)

  def name_for_tensor(self, tensor: _tensor.TensorEntry) -> Tuple[str, RegisteredName]:
    assert self.is_subnetwork()
    assert tensor.names
    return self.name_in_ctx(possible_sub_names=tensor.names, _src_tensor=tensor)

  def find_name_for_module(self, module: _module.ModuleEntry) -> Optional[str]:
    assert self.is_subnetwork()
    for name, child in self.childs_by_name.items():
      if module in child.modules:
        return name
    return None

  def dump(self, prefix=""):
    for name, child in self.childs_by_name.items():
      if name.startswith("."):
        print(f"{prefix}{name}: (hidden, {'non-empty' if child.childs_by_name else 'empty'})")
        continue
      print(f"{prefix}{name}: {child._repr_content()}")
      child.dump(prefix=f"{prefix}  ")

  def dump_as_returnn_layer_dict(self) -> Dict[str, Any]:
    if not self.is_subnet:
      assert len(self.calls) == 1
      call = self.calls[0]
      assert call.returnn_layer_dict
      return call.returnn_layer_dict
    # Subnetwork
    inputs = []
    for input_child in self._inputs:
      input_tensor = input_child.tensor
      assert input_tensor
      assert self.parent
      parent_namespace = self.parent
      input_layer_name, _ = parent_namespace.name_for_tensor(input_tensor)
      inputs.append(input_layer_name)
    subnet_dict = self.dump_as_returnn_net_dict()
    if len(inputs) <= 1:
      return {"class": "subnetwork", "from": inputs[0] if inputs else [], "subnetwork": subnet_dict}
    return {
      "class": "subnetwork", "from": inputs, "concat_sources": False, "subnetwork": subnet_dict}

  def dump_as_returnn_net_dict(self) -> Dict[str, Dict[str, Any]]:
    net_dict = {}
    for name, child in self.childs_by_name.items():
      if not child.calls:
        continue  # e.g. input "data"
      net_dict[name] = child.dump_as_returnn_layer_dict()
      if child.is_alternative_subnet_output:
        net_dict[name]["is_output_layer"] = True
    return net_dict
