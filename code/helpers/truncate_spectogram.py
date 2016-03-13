
class TruncateSpectogram:
    def __init__(self, selector, truncate=150, **kwargs):
        self._selector = selector
        self._truncate = truncate
        self._spectogram_settings = kwargs

    def __iter__(self):
        for item in self._selector:
            spectogram = item.spectogram(**self._spectogram_settings)

            if (spectogram.shape[1] >= self._truncate):
                spectogram = spectogram[:, 0:self._truncate]
                spectogram = spectogram.reshape(1, *spectogram.shape)

                yield (spectogram, int(item.sex == 'f'))
