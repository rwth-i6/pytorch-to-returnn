
from typing import Dict


class ModMap:
  def __init__(self, base_mod_map: Dict[str, str]):
    """
    :param base_mod_map: e.g.
      {"parallel_wavegan": "pytorch_to_returnn._wrapped_mods.parallel_wavegan",
       "torch": "pytorch_to_returnn._wrapped_mods.torch"}
      or
      {"parallel_wavegan": "pytorch_to_returnn._demo_mods.parallel_wavegan",
       "torch": "pytorch_to_returnn.torch"}
    """
    self.base_mod_map = base_mod_map.copy()

  def should_wrap_mod_name(self, mod_name: str) -> bool:
    """
    :param mod_name: e.g. "torch.nn.init"
    :return: whether it should be wrapped / can be mapped. see :func:`map_mod_name`
    """
    if mod_name in self.base_mod_map:
      return True
    for base_mod_name in self.base_mod_map.keys():
      if mod_name.startswith(base_mod_name + "."):
        return True
    return False

  def find_base_mod_prefix(self, mod_name: str) -> str:  # e.g. "torch.nn.init" -> "torch"
    """
    :param mod_name: e.g. "torch.nn.init". :func:`should_wrap_mod_name` must return True
    :return: e.g. "torch"
    """
    if mod_name in self.base_mod_map:
      return mod_name
    for base_mod_name in sorted(self.base_mod_map.keys(), reverse=True):
      if mod_name.startswith(base_mod_name + "."):
        return base_mod_name
    raise Exception(f"cannot map mod name {mod_name!r}. map {self.base_mod_map!r}")

  def map_mod_name(self, mod_name: str) -> str:  # e.g. "torch.nn" -> "pytorch_to_returnn._wrapped_mods.torch.nn"
    """
    :param mod_name: e.g. "torch.nn". :func:`should_wrap_mod_name` must return True
    :return: e.g. "pytorch_to_returnn._wrapped_mods.torch.nn"
    """
    base_mod_name = self.find_base_mod_prefix(mod_name)
    postfix = mod_name[len(base_mod_name):]
    assert postfix.startswith(".") or postfix == ""
    return self.base_mod_map[base_mod_name] + postfix

  def simplify_(self):
    """
    Simplify the internal dict without changing semantics.
    This is done inplace.
    """
    for base_mod_name in sorted(self.base_mod_map.keys(), reverse=True):
      mapped_mod_name = self.base_mod_map.pop(base_mod_name)  # just temporarily
      if not self.should_wrap_mod_name(base_mod_name):
        # Need it, put back in.
        self.base_mod_map[base_mod_name] = mapped_mod_name
      else:
        # Some other entry maps.
        mapped_mod_name_ = self.map_mod_name(base_mod_name)
        if mapped_mod_name != mapped_mod_name_:
          # Need it, put back in.
          self.base_mod_map[base_mod_name] = mapped_mod_name
        else:
          pass  # Can keep it that way.
