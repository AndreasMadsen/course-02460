
import collections

class Filter:
    def __init__(self, selector, min_count=1):
        self._selector = selector
        self._min_count = min_count

        # Count number of observations in each label
        self._class_count = collections.Counter()
        for i, (_, target) in enumerate(selector):
            self._class_count[target] += 1

    def __iter__(self):
        for input, target in self._selector:
            if (self._class_count[target] >= self._min_count):
                yield (input, target)
