
import collections

class CrossValidation:
    def __init__(self, selector, folds=3, stratified=False, **kwargs):
        """
            Corss Validation splits into train and test data.
            If `stratified` is True the split will assure evenly splitted classes.
        """
        self.folds = []
        for fold in range(0, folds):
            self.folds.append(ValidationFold(selector, fold, folds=folds, stratified=stratified))

class ValidationFold:
    def __init__(self, selector, fold, folds, stratified):
        self.selector = selector
        self.fold = fold
        self.folds = folds
        self.stratified = stratified

        if self.stratified:
            self.Splitter = StratifiedFoldSplitter
        else:
            self.Splitter = FoldSplitter

    @property
    def train(self):
        counter = self.Splitter(self.folds)

        for input, target in self.selector:
            if counter.get_and_increment_fold(target) != self.fold:
                yield (input, target)

    @property
    def test(self):
        counter = self.Splitter(self.folds)

        for input, target in self.selector:
            if counter.get_and_increment_fold(target) == self.fold:
                yield (input, target)

class StratifiedFoldSplitter:
    def __init__(self, folds):
        self.folds = folds
        self._next_fold = collections.defaultdict(int)

    def get_and_increment_fold(self, target):
        next_fold = self._next_fold[target]
        self._next_fold[target] = (self._next_fold[target] + 1) % self.folds
        return next_fold

class FoldSplitter:
    def __init__(self, folds):
        self.folds = folds
        self._next_fold = 0

    def get_and_increment_fold(self, target):
        next_fold = self._next_fold
        self._next_fold = (self._next_fold + 1) % self.folds
        return next_fold
