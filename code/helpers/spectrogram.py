
import numpy as np

class Spectrogram:
    def __init__(self, selector, log_transform=False, **kwargs):
        self._selector = selector
        self._log_transform = log_transform
        self._spectrogram_settings = kwargs

    def __iter__(self):
        for item, target in self._selector:
            spectrogram = item.spectrogram(**self._spectrogram_settings)
            spectrogram = spectrogram.reshape(1, *spectrogram.shape)

            if (self._log_transform):
                spectrogram = np.log(1 + 10000 * spectrogram)

            yield (spectrogram, target)

class MeanFrequenciesOfSpectrogram:
    def __init__(self, selector, **kwargs):
        self._selector = selector
        self._spectrogram_settings = kwargs

    def __iter__(self):
        for item in self._selector:
            spectrogram = item.spectrogram(**self._spectrogram_settings)
            mean_freqs  = spectrogram.mean(axis=1)
            mean_freqs  = mean_freqs / np.linalg.norm(mean_freqs)
            yield (mean_freqs, int(item.sex == 'f'))
