
from typing import Set, Iterable, Dict


class WrapCtx:
  def __init__(self, wrapped_mod_prefix: str,
               wrap_mods_direct: Set[str], wrap_mods_indirect: Set[str] = None,
               keep_as_is_types: Iterable[type] = (),
               explicit_wrapped_types: Dict[type, type] = None):
    """
    :param wrapped_mod_prefix: e.g. "pytorch_to_returnn._wrapped_mods."
    :param wrap_mods_direct: e.g. {"torch.nn.modules"}
      These modules will be imported using the direct AST transformation,
      and result in :class:`WrappedSourceModule`.
    :param wrap_mods_indirect: e.g. {"torch", "torch.nn.modules.module"}
      These modules will be wrapped using :class:`WrappedIndirectModule`,
      which wraps the originally unmodified imported module.
      There can be overlaps between wrap_mods_direct and wrap_mods_indirect.
      The topmost entry will decide whether it is direct or indirect.
    :param keep_as_is_types: e.g. {torch.device, torch.dtype, torch.Size}
    :param explicit_wrapped_types: e.g. {}
    """
    assert wrapped_mod_prefix.endswith(".")
    self.wrapped_mod_prefix = wrapped_mod_prefix
    self.wrap_mods_direct = wrap_mods_direct
    if wrap_mods_indirect is None:
      wrap_mods_indirect = set()  # type: Set[str]
    self.wrap_mods_indirect = wrap_mods_indirect
    self.keep_as_is_types = tuple(keep_as_is_types)
    if explicit_wrapped_types is None:
      explicit_wrapped_types = {}  # type: Dict[type, type]
    self.explicit_wrapped_types = explicit_wrapped_types

  def should_wrap_mod(self, mod_name: str) -> bool:
    """
    :param mod_name: e.g. "torch"
    """
    mod_name = mod_name.split(".")
    for i in range(1, len(mod_name) + 1):
      if ".".join(mod_name[:i]) in self.wrap_mods_direct:
        return True
      if ".".join(mod_name[:i]) in self.wrap_mods_indirect:
        return True
    return False


def make_torch_default_ctx(wrapped_mod_prefix: str) -> WrapCtx:
  import torch
  _KeepAsIsTypes = (
    torch.device,
    torch.Size,
    torch.dtype,
  )
  from .torch_wrappers import WrappedTorchTensor, WrappedTorchParameter, WrappedModuleBase
  _TypeMap = {
    torch.Tensor: WrappedTorchTensor,
    torch.nn.Parameter: WrappedTorchParameter,
    torch.nn.Module: WrappedModuleBase,
  }
  return WrapCtx(
    wrapped_mod_prefix=wrapped_mod_prefix,
    wrap_mods_direct=_TorchExplicitDirectModList,
    wrap_mods_indirect=_TorchExplicitIndirectModList,
    keep_as_is_types=_KeepAsIsTypes,
    explicit_wrapped_types=_TypeMap)


_TorchExplicitIndirectModList = {
  # Leave "torch" in to not do any magic on the original Torch modules.
  # Note that the main torch module cannot be transformed
  # because of some side-effects triggered by native Torch code,
  # which we cannot catch automatically.
  # More specifically, in torch/__init__.py, there is the call _C._initExtension,
  # which (un-intuitively) does all sorts of initialization,
  # including torch::utils::initializeDtypes,
  # which itself has the side-effect of adding dtypes (e.g. `float32`)
  # to the `torch` module (-> `torch.float32`).
  # And the name `"torch"` is hardcoded there,
  # i.e. it will not used our wrapped module,
  # thus we will never have `float32` registered,
  # and that leads to exceptions.
  # This is likely not the only problem.
  "torch",
}
if "torch" in _TorchExplicitIndirectModList:
  _TorchExplicitIndirectModList.update({
    # If we would transform `torch` directly, the following should be explicitly excluded.
    "torch.tensor", "torch.distributed",
    "torch._C",
    # Avoid to have the `Module` class twice.
    "torch.nn.modules.module",
  })

_TorchExplicitDirectModList = {
  # Note: When we don't wrap all of `torch` (i.e. `"torch"` is in _ExplicitIndirectModList),
  # this will partially be WrappedIndirectModule and WrappedSourceModule.
  # This is not a problem in principle, but there can be subtle problems.
  # Also the import order can be important.
  # E.g. "torch.nn.modules" should be imported before "torch.nn.functional".
  # This might need extra explicit handling, depending what we have here in this list.
  "torch.nn.modules",
  # "torch.nn.functional",  # -- not needed, also causes other problems
}
