
import numpy as np

class TargetType:
    def __init__(self, iterable, target_type='sex'):
        self._iterable = iterable
        self._target_type = target_type
        self._init_target_function(target_type)

    def __iter__(self):
        for item in self._iterable:
            yield item, self._target_function(item)

    def _init_target_function(self, target_type):
        # TODO(Lasse): This can be generalized to
        # not be dependent of `target_type`.
        if (target_type == 'sex'):
            self._sex_types = {'m': 0, 'f': 1}
            self._target_function = self._sex_target
        elif (target_type == 'speaker'):
            # Gather all speakers in a dictionary
            speaker_types = sorted(list(set([item.speaker for item in self._iterable])))
            self._speakers = { speaker: idx for idx, speaker in enumerate(speaker_types)}
            self._target_function = self._speaker_target
        else:
            raise ValueError('target_type could not be found!')

    def _sex_target(self, item):
        """ Returns 0 for male and 1 for female. """
        return self._sex_types[item.sex]

    def _speaker_target(self, item):
        """ Returns one-hit vector for a given speaker. """
        return self._speakers[item.speaker]

    @property
    def labels(self):
        if (self._target_type == 'sex'):
            return self._sex_types
        elif (self._target_type == 'speaker'):
            return self._speakers
