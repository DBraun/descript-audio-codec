import torch
from audiotools import AudioSignal
import argbind
import dac


@argbind.bind(without_prefix=True)
def benchmark_dac(model_type="44khz", model_bitrate='8kbps', win_durations=None):

    model_path = dac.utils.download(model_type=model_type, model_bitrate=model_bitrate)
    model = dac.DAC.load(model_path)

    model = model.to("cuda")

    print(f'Benchmarking model: {model_type}, {model_bitrate}')

    if win_durations is None:
        win_durations = [0.37, 0.38, 0.42, 0.46, 0.5, 1, 5, 10, 20]

    with torch.no_grad():

        # the length doesn't matter since it will be padded anyway.
        x = AudioSignal(torch.randn(1, 1, 44100), model.sample_rate)
        x = x.to('cuda')

        for win_duration in win_durations:
            print('win_duration', win_duration)
            dac_file = model.compress(x, win_duration=win_duration, verbose=False, benchmark=True)
            model.decompress(dac_file, verbose=False, benchmark=True)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        benchmark_dac()
