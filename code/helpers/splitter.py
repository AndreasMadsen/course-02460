
class Splitter:
    def __init__(self, selector, split_size, axis=1):
        self._selector   = selector
        self._split_size = split_size
        self._axis = axis

        # Builds a slicer that is used for splitting the input.
        dims = 0
        for input, target in selector:
            dims = input.ndim
            break
        self._slicer = [slice(None)] * dims

    def _slice(self, idx):
        slicer = list(self._slicer)
        slicer[self._axis] = slice(idx, idx + self._split_size)
        return slicer

    def __iter__(self):
        for input, target in self._selector:
            for idx in range(
                0,
                input.shape[self._axis] - self._split_size,
                self._split_size
            ):
                yield (input[self._slice(idx)], target)
