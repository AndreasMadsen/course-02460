import sys
import numpy as np

import timit
import helpers

from sklearn import grid_search
from sklearn.svm import SVC

def main():

    # File selectors
    train_selector = helpers.MeanFrequenciesOfSpectrogram(
        timit.FileSelector(usage='train', dialect=None))
    train_iterable = helpers.Minibatch(train_selector, cache=True)

    test_selector = helpers.MeanFrequenciesOfSpectrogram(
        timit.FileSelector(usage='test', dialect=None))
    test_iterable = helpers.Minibatch(test_selector, cache=True)

    X_train, Y_train = train_iterable.data
    X_test, Y_test = test_iterable.data

    # Parameter grid
    param_grid = {
        'C': [60000, 61000, 62000, 63000, 64000, 65000]
    }

    model = SVC(kernel='rbf')

    # Initialize model
    clf = grid_search.GridSearchCV(model, param_grid)

    # Fit model
    clf.fit(X_train, Y_train)

    # Print best scorer
    print('Best estimator:')
    print(clf.best_estimator_)
    print('Best parameters:')
    print(clf.best_params_)
    print('Best CV score:')
    print(clf.best_score_)

    print('Test score:')
    print(clf.best_estimator_.score(X_test, Y_test))

    # Best estimator is obtained using
    # {'kernel': 'rbf', 'C': 60000}
    # Accuracy: 0.975714285714

if __name__ == '__main__':
    main()
