
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


from sklearn.utils.extmath import logsumexp
from sklearn import mixture

# Create data selector object
selector = elsdsr.FileSelector()
#selector = timit.FileSelector()
#selector = helpers.Filter(selector, target='speaker', min_count=10, min_size=300, nperseg=256, noverlap=128)
selector = helpers.TargetType(selector, target='speaker')
speakers = selector.labels
selector = helpers.MFCC(selector, normalize_signal=True)
selector = helpers.Splitter(selector, split_size=100, axis=0)
#selector = helpers.Truncate(selector, truncate=300, axis=0)
selector = helpers.Validation(selector, test_fraction=0.25, stratified=True)

speakers = range(0, len(speakers))

X_train = collections.defaultdict(list)
for input, target in selector.train:
    X_train[target].append(input.ravel())

models = []
for speaker in speakers:
    clf = GMM(n_components=3)
    clf.fit(X_train[speaker])
    models.append(clf)

print('Speaker count: %d' % (len(speakers)))

errors = 0
observations = 0
for input, target in selector.test:
    probs = np.array([model.score([input.ravel()])[0] for model in models])
    y_hat = np.argmax(probs)
    errors += (y_hat != target)
    #print('y_hat = %d == %d' % (y_hat, target))
    observations += 1


MSE = errors / observations
print('missrate = %.4f' % (MSE))
