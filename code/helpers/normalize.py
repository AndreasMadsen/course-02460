
import math

import numpy as np

class Normalize:
    def __init__(self, iterable):
        self._iterable = iterable

        # Estimate mean and variance using the online Knuth-Welford algorithm
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
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

    def __iter__(self):
        for input, target in self._iterable:
            yield ((input - self.mean) / self.stddev, target)
