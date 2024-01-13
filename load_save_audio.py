import scipy.io.wavfile as wavfile
import scipy.signal as signal

import torch
import numpy as np

def load_data(path='music_data_folder/1.wav'):
    samplerate, data = wavfile.read(path)

    samplerate_new = 4096
    data_resampled = signal.resample(data, samplerate_new * len(data) // samplerate)

    data_mono = (data_resampled[:, 0] + data_resampled[:, 1]) / 2
    scale = int((data_mono.max() - data_mono.min()) / (512*2))

    ret = torch.from_numpy((data_mono / scale)).to(torch.int16)
    return ret, samplerate_new, scale

def save_audio(audio_tensor, sample_rate, audio_path, scale=64):
    audio_numpy = audio_tensor.numpy() * scale
    wavfile.write(audio_path, sample_rate, audio_numpy.astype(np.int16))