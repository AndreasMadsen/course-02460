
import collections

class Validation:
    def __init__(self, selector, test_fraction=0.33, stratified=False, **kwargs):
        """
            Splits selector into train and test data.
            If `stratified` is True the split will assure evenly splitted classes.
        """
        self._selector = selector

        # Create idx splitting mapping
        # 0 = Train
        # 1 = Test
        self._split_idx = list()

        # TODO (Andreas): I'm pretty sure these for loops can be moved inside
        # the train and test methods, since the selector order is deterministic.
        if stratified:
            # Count labels in each subset
            in_test = collections.Counter()
            in_train = collections.Counter()

            # Divide dataset
            for i, (input, target) in enumerate(self._selector):
                if self._test_ratio(in_test[target], in_train[target]) < test_fraction:
                    self._split_idx.append(1)
                    in_test[target] += 1
                else:
                    self._split_idx.append(0)
                    in_train[target] += 1
        else:
            # Count labels in each subset
            in_test = 0
            in_train = 0

            # Divide dataset
            for i, _ in enumerate(self._selector):
                if self._test_ratio(in_test, in_train) < test_fraction:
                    self._split_idx.append(1)
                    in_test += 1
                else:
                    self._split_idx.append(0)
                    in_train += 1

    @staticmethod
    def _test_ratio(in_test, in_train):
        if in_test + in_train == 0: return 0
        return in_test / (in_test + in_train)

    @property
    def train(self):
        for i, (input, target) in enumerate(self._selector):
            if self._split_idx[i] == 0:
                yield (input, target)

    @property
    def test(self):
        for i, (input, target) in enumerate(self._selector):
            if self._split_idx[i] == 1:
                yield (input, target)
