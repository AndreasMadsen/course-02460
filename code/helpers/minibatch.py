
import numpy as np

class Minibatch:
    def __init__(self, data_iterable, batchsize=50,
                 input_type='float32', target_type='int32'):
        self._data_iterable = data_iterable
        self._batchsize = 50
        self._input_type = input_type
        self._target_type = target_type

        input_data = []
        target_data = []
        for input, target in data_iterable:
            input_data.append(input)
            target_data.append(target)

        self._input_cache = np.asarray(input_data, dtype=self._input_type)
        self._target_cache = np.asarray(target_data, dtype=self._target_type)

    def __iter__(self):
        return MinibatchCache(self._input_cache, self._target_cache,
                              self._batchsize)

    @property
    def data(self):
        return (self._input_cache, self._target_cache)

class MinibatchCache:
    def __init__(self, input_cache, target_cache, batchsize):
        self._input_cache = input_cache
        self._target_cache = target_cache

        self._size = input_cache.shape[0]
        self._batchsize = batchsize

        self._order = np.random.permutation(self._size)
        self._position = 0

    def __next__(self):
        if (self._position >= self._size): raise StopIteration

        curr_position = self._position
        next_position = self._position + self._batchsize

        # Select batch by only copying the data subset
        input_batch = self._input_cache[self._order[curr_position:next_position]]
        target_batch = self._target_cache[self._order[curr_position:next_position]]

        # Move position for next iteration
        self._position = next_position

        return (input_batch, target_batch)
