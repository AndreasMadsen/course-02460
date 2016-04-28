
import collections

class Filter:
    def __init__(self, selector, min_count=1, target=None, min_size=None, *args, **kwargs):
        self._selector = selector

        if (min_count > 1 and target is None):
            raise ValueError('target must be specified when min_count is used')

        self._target = target
        self._min_count = min_count
        self._min_size = min_size

        self._args = args
        self._kwargs = kwargs

        # Count number of observations in each label
        self._class_count = collections.Counter()
        if min_count > 1:
            for item in selector:
                if self._size_valid(item):
                    self._class_count[getattr(item, target)] += 1

    def _size_valid(self, item):
        if self._min_size is None: return True
        return item.size(*self._args, **self._kwargs) >= self._min_size

    def _count_valid(self, item):
        if self._min_count == 1: return True
        return self._class_count[getattr(item, self._target)] >= self._min_count

    def __iter__(self):
        for item in self._selector:
            if self._count_valid(item) and self._size_valid(item):
                yield item
