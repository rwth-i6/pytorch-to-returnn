
import ast
import linecache
from .. import log
from typing import Optional, Set


class AstImportTransformer(ast.NodeTransformer):
  def __init__(self, base_mod_names: Set[str], new_import_prefix: str, src_filename: str):
    """
    :param base_mod_names: e.g. {"parallel_wavegan", "torch"}
    :param new_import_prefix: e.g. "pytorch_to_returnn._wrapped_mods."
    :param src_filename: just for logging
    """
    self.base_mod_names = base_mod_names
    self.new_import_prefix = new_import_prefix
    assert self.new_import_prefix.endswith(".")
    self.src_filename = src_filename

  def _should_wrap_mod_name(self, mod_name: str) -> bool:
    if mod_name in self.base_mod_names:
      return True
    for base_mod_name in self.base_mod_names:
      if mod_name.startswith(base_mod_name + "."):
        return True
    return False

  def visit_Import(self, node: ast.Import):
    # https://docs.python.org/3/library/ast.html#ast.Import
    if not any([self._should_wrap_mod_name(alias.name) for alias in node.names]):
      return node
    if len(node.names) >= 2:
      # Break down into multiple imports.
      res = []
      for alias in node.names:
        assert isinstance(alias, ast.alias)
        sub_node = ast.Import(names=[alias])
        ast.copy_location(sub_node, node)
        sub_node = self.visit_Import(sub_node)
        assert isinstance(sub_node, ast.Import)
        res.append(sub_node)
      return res
    assert len(node.names) == 1
    alias, = node.names
    assert isinstance(alias, ast.alias)
    if not self._should_wrap_mod_name(alias.name):
      return node
    if log.Verbosity >= 10:
      log.unique_print("*** AST transform %r (%s)" % (_ast_get_source_segment(self.src_filename, node), ast.dump(node)))
    # E.g. `import parallel_wavegan.layers`
    # ->
    # `import pytorch_to_returnn._wrapped_mods.parallel_wavegan.layers`
    # If "." in mod name, also:
    # `from pytorch_to_returnn._wrapped_mods import parallel_wavegan`
    # If `import parallel_wavegan.layers as x`, simpler:
    if alias.asname:  # import torch as x -> import pytorch_to_returnn._wrapped_mods.torch as x
      new_node = ast.Import(
        names=[ast.alias(
          name=self.new_import_prefix + alias.name, asname=alias.asname)])
      ast.copy_location(new_node, node)
      return new_node
    # If "." in mod name, simpler:
    if "." not in alias.name:  # import torch -> import pytorch_to_returnn._wrapped_mods.torch as torch
      new_node = ast.Import(
        names=[ast.alias(
          name=self.new_import_prefix + alias.name, asname=alias.name)])
      ast.copy_location(new_node, node)
      return new_node
    assert not alias.asname
    new_import_node = ast.Import(names=[ast.alias(name=self.new_import_prefix + alias.name)])
    ast.copy_location(new_import_node, node)
    assert "." in alias.name
    base_mod_name = alias.name.partition(".")[0]
    new_alias_node = ast.ImportFrom(  # from pytorch_to_returnn._wrapped_mods import torch
      module=self.new_import_prefix[:-1], names=[ast.alias(name=base_mod_name)])
    ast.copy_location(new_alias_node, node)
    return [new_import_node, new_alias_node]

  def visit_ImportFrom(self, node: ast.ImportFrom):
    # https://docs.python.org/3/library/ast.html#ast.ImportFrom
    if node.level == 0 and not self._should_wrap_mod_name(node.module):
      return node
    if node.level > 0:
      return node  # relative imports should already be handled correctly
    assert node.level == 0
    if log.Verbosity >= 10:
      log.unique_print("*** AST transform %r (%s)" % (_ast_get_source_segment(self.src_filename, node), ast.dump(node)))
    new_node = ast.ImportFrom(
      module=self.new_import_prefix + node.module, names=node.names, level=0)
    ast.copy_location(new_node, node)
    return new_node


def _ast_get_source_segment(src_filename: str, node: ast.AST) -> Optional[str]:
    """Get source code segment of the *source* that generated *node*.

    If some location information (`lineno`, `end_lineno`, `col_offset`,
    or `end_col_offset`) is missing, return None.

    Original ast.get_source_segment is very inefficient. This is faster.
    """
    try:
        lineno = node.lineno - 1
        end_lineno = getattr(node, "end_lineno", node.lineno) - 1
        col_offset = getattr(node, "col_offset", 0)
        end_col_offset = getattr(node, "end_col_offset", -1)
    except AttributeError:
        return None

    lines = linecache.getlines(src_filename)
    if end_lineno == lineno:
        return lines[lineno].encode()[col_offset:end_col_offset].decode()

    first = lines[lineno].encode()[col_offset:].decode()
    last = lines[end_lineno].encode()[:end_col_offset].decode()
    lines = lines[lineno+1:end_lineno]

    lines.insert(0, first)
    lines.append(last)
    return ''.join(lines)
