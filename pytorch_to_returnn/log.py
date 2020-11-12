

Verbosity = 0


_unique_prints = set()


def unique_print(txt: str):
  if Verbosity >= 100:  # always print, ignore unique
    print(txt)
    return
  if txt in _unique_prints:
    return
  _unique_prints.add(txt)
  print(txt)
