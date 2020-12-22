Note that this is about the current state of the project.
We can potentially find solutions for all of this.
It was just not needed so far.

Also see the list of [supported modules and functions](Supported.md).


# Inplace ops

PyTorch supports inplace operations,
e.g. most functions with `_` postfix,
but also the standard Python inplace-ops (`+=` etc).
TensorFlow does not support this.
You still can easily get an equivalent op in all cases
(`+=` works just fine, but just creates a new tensor)
but we would need to be careful
to retain the same behavior
(e.g. `y = x; x += 1; print(y)`).
This is possible, but complicated.


# `numpy` and `from_numpy`

`from_numpy` currently assumes
that the data is an independent constant,
i.e. it will not be part of the computation,
and in the translation,
it will just be used as constant.

We could also provide a symbolic wrapper
for `numpy.ndarray`,
just as we have for `torch.Tensor`,
and provide very similar logic.
But this is not done yet,
and probably not needed in most cases.
