
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
#selector = timit.FileSelector()
selector = elsdsr.FileSelector()
#selector = helpers.Filter(selector, target='speaker', min_count=10, min_size=300, nperseg=256, noverlap=128)
selector = helpers.TargetType(selector, target='speaker')
speakers = selector.labels
selector = helpers.MFCC(selector, normalize_signal=True)
#selector = helpers.Truncate(selector, truncate=100, axis=0)
selector = helpers.Splitter(selector, split_size=100, axis=0)
#selector = helpers.Validation(selector, test_fraction=0.25, stratified=True)
selector = helpers.Validation(selector, test_fraction=0.50, stratified=True)

speakers = list(range(0, len(speakers)))

X_train = collections.defaultdict(list)
for input, target in selector.train:
    X_train[target].append(input.ravel())



X_test = collections.defaultdict(list)
for input, target in selector.test:
    X_test[target].append(input.ravel())

for key, values in X_train.items():
    X_train[key] = np.array(values)

for key, values in X_test.items():
    X_test[key] = np.array(values)

models = []
for speaker in sorted(speakers):
    #clf = GMM(n_components=5) # TODO fix number of samples
    clf = GMM(n_components=3)
    clf.fit(X_train[speaker])
    models.append(clf)

errors = 0
observations = 0
for input, target in selector.test:
    #for model in models:
    #    prob = model.predict_proba([input.ravel()])
    #    print('prob')
    #    for x in prob[0]:
    #        print(x)
    #    print('')

    #import sys
    #sys.exit()
    probs = np.array([model.predict_proba([input.ravel()])[0] for model in models])
    #print(probs)
    #print(probs.shape)
    y_hat = np.argmax(probs)
    errors += (y_hat != target)
    print('y_hat = %d == %d' % (y_hat, target))
    observations += 1

MSE = errors / observations
print('missrate = %.4f' % (MSE))
