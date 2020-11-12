

class _DummyLoad:
  def __getitem__(self, item):
    return self


def load(*args, **kwargs):
  return _DummyLoad()
