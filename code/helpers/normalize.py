
import math

import numpy as np

class Normalize:
    def __init__(self, iterable, mean=None, stddev=None):
        self._iterable = iterable

        if mean is None or stddev is None:
            inputs = np.asarray([input for input, target in iterable])

        if mean is None:
            self.mean = np.mean(inputs.ravel())
        else:
            self.mean = mean

        if stddev is None:
            self.stddev = np.std(inputs.ravel())
        else:
            self.stddev = stddev

        # Estimate mean and variance using the online Knuth-Welford algorithm
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        # TODO: estimate mean and stddev using Knuth-Welford to prevent numerical
        # errors and avoid buffering of all the data.
        """
        n = 0
        mean = 0.0
        M2 = 0.0

        for input, target in iterable:
            x = np.mean(input.ravel())
            n += 1
            delta = x - mean
            mean += delta / n
            M2 += delta * (x - mean)

        self.observations = n
        self.mean = mean
        self.variance = M2 / (n - 1)
        self.stddev = math.sqrt(self.variance)
        """

    def __iter__(self):
        for input, target in self._iterable:
            yield ((input - self.mean) / self.stddev, target)
