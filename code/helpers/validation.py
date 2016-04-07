
import numpy as np

class Validation:
    def __init__(self, selector, test_fraction=0.33, **kwargs):
        self.selector = selector
        n = sum([1 for _ in selector])
        self.indices = np.arange(n)
        np.random.shuffle(self.indices)
        self.n_split = int(n * test_fraction)

    @property
    def train(self):
        for i, (input, target) in enumerate(self.selector):
            if (self.indices[i] >= self.n_split):
                yield input, target

    @property
    def test(self):
        for i, (input, target) in enumerate(self.selector):
            if (self.indices[i] < self.n_split):
                yield input, target
