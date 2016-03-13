
class TruncateSpectrogram:
    def __init__(self, selector, truncate=150, **kwargs):
        self._selector = selector
        self._truncate = truncate
        self._spectrogram_settings = kwargs

    def __iter__(self):
        for item in self._selector:
            spectrogram = item.spectrogram(**self._spectrogram_settings)

            if (spectrogram.shape[1] >= self._truncate):
                spectrogram = spectrogram[:, 0:self._truncate]
                spectrogram = spectrogram.reshape(1, *spectrogram.shape)

                yield (spectrogram, int(item.sex == 'f'))
