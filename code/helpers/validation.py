
import collections

class Validation:
    def __init__(self, selector, test_fraction=0.33, stratified=False, **kwargs):
        """
            Splits selector into train and test data.
            If `stratified` is True the split will assure evenly splitted classes.
        """
        self._selector = selector
        self._test_fraction = test_fraction
        self._stratified = stratified

    @property
    def train(self):
        counter = SeperationCounter(self._stratified)

        for input, target in self._selector:
            if counter.test_ratio(target) >= self._test_fraction:
                counter.increment_train(target)
                yield (input, target)
            else:
                counter.increment_test(target)

    @property
    def test(self):
        counter = SeperationCounter(self._stratified)

        for input, target in self._selector:
            if counter.test_ratio(target) < self._test_fraction:
                counter.increment_test(target)
                yield (input, target)
            else:
                counter.increment_train(target)

class SeperationCounter:
    def __init__(self, stratified):
        self._stratified = stratified

        if self._stratified:
            # Count labels in each subset
            self._in_test = collections.Counter()
            self._in_train = collections.Counter()
        else:
            # Count labels in each subset
            self._in_test = 0
            self._in_train = 0

    def test_ratio(self, target):
        in_test = self._in_test[target] if self._stratified else self._in_test
        in_train = self._in_train[target] if self._stratified else self._in_train

        if in_test + in_train == 0: return 0
        return in_test / (in_test + in_train)

    def increment_train(self, target):
        if self._stratified:
            self._in_train[target] += 1
        else:
            self._in_train += 1

    def increment_test(self, target):
        if self._stratified:
            self._in_test[target] += 1
        else:
            self._in_test += 1
