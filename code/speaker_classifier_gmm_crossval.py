
import os
import numpy as np
import lasagne
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import collections
from sklearn.mixture import GMM

import timit
import elsdsr
import network
import helpers
import plot
import early_stopping

# Create data selector object
selector = timit.FileSelector()
selector = helpers.Filter(selector, target='speaker', min_count=10, min_size=300, nperseg=256, noverlap=128)
selector = helpers.TargetType(selector, target='speaker')
speakers = selector.labels
selector = helpers.MFCC(selector, normalize_signal=True)
selector = helpers.Splitter(selector, split_size=100, axis=0)
selector = helpers.Normalize(selector)
selector = helpers.CrossValidation(selector, folds=5, stratified=True)

missclassifications_list = []
speakers = range(0, len(speakers))

for fold in selector.folds:
    X_train = collections.defaultdict(list)
    for input, target in fold.train:
        X_train[target].append(input.ravel())

    models = []
    for speaker in speakers:
        clf = GMM(n_components=3)
        clf.fit(X_train[speaker])
        models.append(clf)

    errors = 0
    observations = 0
    for input, target in fold.test:
        probs = np.array([model.score([input.ravel()])[0] for model in models])
        errors += (np.argmax(probs) != target)
        observations += 1

    missclassifications_list.append(errors / observations)
    print('missrate: %f' % (errors / observations))

print(missclassifications_list)
