
import numpy as np

class Minibatch:
    def __init__(self, data_iterable, batchsize=50, cache=False,
                 input_type='float32', target_type='int32'):
        self._data_iterable = data_iterable
        self._batchsize = 50
        self._cache = cache
        self._input_type = input_type
        self._target_type = target_type

        if (cache):
            (input_data, target_data) = zip(*data_iterable)
            self._input_cache = np.asarray(input_data, dtype=self._input_type)
            self._target_cache = np.asarray(target_data, dtype=self._target_type)

    def __iter__(self):
        if (self._cache is False):
            return MinibatchLazy(self._data_iterable, self._batchsize,
                                 self._input_type, self._target_type)
        else:
            return MinibatchCache(self._input_cache, self._target_cache,
                                  self._batchsize)

    @property
    def data(self):
        X, Y = [], []
        for x, y in self._data_iterable:
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y).squeeze()

class MinibatchLazy:
    def __init__(self, data_iterable, batchsize, input_type, target_type):
        self._data_iterator = iter(data_iterable)
        self._batchsize = batchsize
        self._input_type = input_type
        self._target_type = target_type
        self._stop = False

    def __next__(self):
        if (self._stop): raise StopIteration

        # Collect `batchsize` items and set _stop if there are no more items
        input_batch = []
        target_batch = []
        while len(input_batch) < self._batchsize:
            try:
                (input, target) = next(self._data_iterator)
                input_batch.append(input)
                target_batch.append(target)
            except StopIteration:
                self._stop = True
                break

        # If there are no items just stop immediately
        if (len(input_batch) == 0): raise StopIteration

        # Output `batchsize` items or whatever is remaining
        return (
            np.asarray(input_batch, dtype=self._input_type),
            np.asarray(target_batch, dtype=self._target_type)
        )


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
