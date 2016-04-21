
class Binary:
    def __init__(self, iterable):
        self._iterable = iterable

        for _, target in self._iterable:
            self._binary_label = target
            break

    def __iter__(self):
        for input, target in self._iterable:
            yield input, int(target == self._binary_label)
