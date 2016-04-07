
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
        if (target_type == 'sex'):
            self._target_function = self._sex_target
        elif (target_type == 'speaker'):
            # Gather all speakers in an array
            self._speakers = np.array(list(set([item.speaker for item in self._iterable])))
            self._target_function = self._speaker_target
        else:
            raise ValueError('target_type could not be found!')

    def _sex_target(self, item):
        """ Returns 0 for male and 1 for female. """
        return int(item.sex == 'f')

    def _speaker_target(self, item):
        """ Returns one-hit vector for a given speaker. """
        return int(np.where(self._speakers == item.speaker)[0])
        #return (self._speakers == item.speaker).astype('int')
