
import collections

class Filter:
    def __init__(self, selector, min_count=1, min_size=None, *args, **kwargs):
        self._selector = selector

        self._min_count = min_count
        self._min_size = min_size

        self._args = args
        self._kwargs = kwargs

        # Count number of observations in each label
        self._class_count = collections.Counter()
        for item, target in selector:
            if self._size_valid(item):
                self._class_count[target] += 1

    def _size_valid(self, item):
        if self._min_size is None: return True
        return item.size(*self._args, **self._kwargs) >= self._min_size

    def __iter__(self):
        for item, target in self._selector:
            if self._class_count[target] >= self._min_count and self._size_valid(item):
                yield (item, target)
