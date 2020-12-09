#!/usr/bin/env python3

"""
Usage example::

  python3 pytorch_to_returnn.py \
  --pwg_config mb_melgan.v2.yaml \
  --pwg_checkpoint mb_melgan_models/checkpoint-1000000steps.pkl \
  --features data/features.npy

"""

import _setup_env  # noqa
import argparse
import torch
import numpy
import yaml
import wave
import better_exchook
import typing
from returnn.tf.util.basic import debug_register_better_repr, setup_tf_thread_pools, print_available_devices
from returnn.log import log


def main():
    parser = argparse.ArgumentParser(description="MB-MelGAN vocoder")
    parser.add_argument("--features", required=True, help="npy file. via decoder.py --dump_features")
    parser.add_argument("--pwg_config", type=str, help="ParallelWaveGAN config (.yaml)")
    parser.add_argument("--pwg_checkpoint", type=str, help="ParallelWaveGAN checkpoint (.pkl)")
    args = parser.parse_args()

    better_exchook.install()
    debug_register_better_repr()
    log.initialize(verbosity=[5])
    setup_tf_thread_pools()
    print_available_devices()

    def model_func(wrapped_import, inputs: torch.Tensor):
        if typing.TYPE_CHECKING or not wrapped_import:
            import torch
            from parallel_wavegan import models as pwg_models
            from parallel_wavegan import layers as pwg_layers

        else:
            torch = wrapped_import("torch")
            wrapped_import("parallel_wavegan")
            pwg_models = wrapped_import("parallel_wavegan.models")
            pwg_layers = wrapped_import("parallel_wavegan.layers")

        # Initialize PWG
        pwg_config = yaml.load(open(args.pwg_config), Loader=yaml.Loader)
        pyt_device = torch.device("cpu")
        generator = pwg_models.MelGANGenerator(**pwg_config['generator_params'])
        generator.load_state_dict(
            torch.load(args.pwg_checkpoint, map_location="cpu")["model"]["generator"])
        generator.remove_weight_norm()
        pwg_model = generator.eval().to(pyt_device)
        assert pwg_config["generator_params"].get("aux_context_window", 0) == 0  # not implemented otherwise
        pwg_pqmf = pwg_layers.PQMF(pwg_config["generator_params"]["out_channels"]).to(pyt_device)

        with torch.no_grad():
            return pwg_pqmf.synthesis(pwg_model(inputs))

    feature_data = numpy.load(args.features)
    print("Feature shape:", feature_data.shape)

    import pytorch_to_returnn.log
    pytorch_to_returnn.log.Verbosity = 6
    from pytorch_to_returnn.converter import verify_torch_and_convert_to_returnn
    verify_torch_and_convert_to_returnn(model_func, inputs=feature_data[None, :, :])
    # from pytorch_to_returnn.wrapped_import import wrapped_import_demo
    # from pytorch_to_returnn import torch as torch_returnn
    # model_func(wrapped_import_demo, inputs=torch_returnn.from_numpy(feature_data[None, :, :]))

    audio_waveform = model_func(None, inputs=torch.from_numpy(feature_data[None, :, :]))
    audio_waveform = audio_waveform.view(-1).cpu().numpy()
    audio_raw = numpy.asarray(audio_waveform*(2**15-1), dtype="int16").tobytes()

    out_fn = "out.wav"
    wave_writer = wave.open(out_fn, "wb")
    wave_writer.setnchannels(1)
    wave_writer.setframerate(16000)
    wave_writer.setsampwidth(2)
    wave_writer.writeframes(audio_raw)
    wave_writer.close()
    print("Wrote %s." % out_fn)


if __name__ == "__main__":
    main()
