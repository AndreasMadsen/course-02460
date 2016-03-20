
import numpy as np

class Spectrogram:
    def __init__(self, selector, log_transform=False, **kwargs):
        self._selector = selector
        self._log_transform = log_transform
        self._spectrogram_settings = kwargs

    def __iter__(self):
        for item in self._selector:
            spectrogram = item.spectrogram(**self._spectrogram_settings)
            spectrogram = spectrogram.reshape(1, *spectrogram.shape)

            if (self._log_transform):
                spectrogram = np.log(1 + 10000 * spectrogram)

            yield (spectrogram, int(item.sex == 'f'))
