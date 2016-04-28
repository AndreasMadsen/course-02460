
import collections

class TargetType:
    def __init__(self, iterable, target='sex'):
        self._iterable = iterable

        self._target = target
        self._label_2_id = self._init_label_dict()

    def __iter__(self):
        for item in self._iterable:
            yield item, self._get_idx(item)

    def _get_idx(self, item):
        label = getattr(item, self._target)
        return self._label_2_id[label]

    def _init_label_dict(self):
        labels = set()

        for item in self._iterable:
            label = getattr(item, self._target)
            labels.add(label)

        return collections.OrderedDict({
            label: idx for idx, label in enumerate(sorted(labels))
        })

    @property
    def labels(self):
        return self._label_2_id.keys()
