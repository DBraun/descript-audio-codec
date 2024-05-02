from typing import List

import argbind
import torch
from audiotools import AudioSignal
import dac


@argbind.bind(without_prefix=True)
def benchmark_dac(model_type="44khz", model_bitrate='8kbps', win_durations: List[str] = None):

    if win_durations is None:
        win_durations = [0.37, 0.38, 0.42, 0.46, 0.5, 1, 5, 10, 20]
    else:
        win_durations = [float(x) for x in win_durations]

    model_path = dac.utils.download(model_type=model_type, model_bitrate=model_bitrate)
    model = dac.DAC.load(model_path)

    model = model.to("cuda")

    print(f'Benchmarking model: {model_type}, {model_bitrate}')

    with torch.no_grad():
        for win_duration in win_durations:
            # force chunk-encoding by making duration 1 more than win_duration:
            T = 1+int(win_duration*model.sample_rate)
            x = AudioSignal(torch.randn(1, 1, T), model.sample_rate)
            x = x.to('cuda')
            try:
                dac_file = model.compress(x, win_duration=win_duration, verbose=False, benchmark=True)
                model.decompress(dac_file, verbose=False, benchmark=True)
            except Exception as e:
                print(f'Exception for win duration "{win_duration}": {e}')


if __name__ == "__main__":
    # example usage:
    # python3 benchmark.py --model_type 16khz --win_durations "0.5 1 5 10 20"
    print('Device count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)
    args = argbind.parse_args()
    with argbind.scope(args):
        benchmark_dac()
