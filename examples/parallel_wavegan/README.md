This wraps [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN).

You find some dependencies (scripts) [here](https://github.com/rwth-i6/returnn-experiments/tree/master/2020-TTS-LJSpeech).
Specifically, copy the `scripts` directory from there to here.
Also, currently we have not covered the training part here,
so you might want to run the given pipeline anyway,
to create your own PyTorch model checkpoint.

This example was earlier in its own repo [here](https://github.com/albertz/import-parallel-wavegan).

See [pytorch_to_returnn.py](pytorch_to_returnn.py).
