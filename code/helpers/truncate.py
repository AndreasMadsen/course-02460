
class Truncate:
    def __init__(self, selector, truncate, axis=1):
        self._selector = selector
        self._truncate = truncate
        self._axis = axis

        # Builds a slicer that truncates `axis` at `truncate`
        dims = 0
        for input, target in selector:
            dims = input.ndim
            break
        self._slicer = [slice(None)] * dims
        self._slicer[axis] = slice(0, truncate)

    def __iter__(self):
        for input, target in self._selector:
            if (input.shape[self._axis] >= self._truncate):
                yield (input[self._slicer], target)
