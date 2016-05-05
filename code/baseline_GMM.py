
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
#selector = timit.FileSelector()
selector = elsdsr.FileSelector()
#selector = helpers.Filter(selector, target='speaker', min_count=10, min_size=300, nperseg=256, noverlap=128)
selector = helpers.TargetType(selector, target='speaker')
speakers = selector.labels
selector = helpers.MFCC(selector, normalize_signal=True)
#selector = helpers.Truncate(selector, truncate=100, axis=0)
selector = helpers.Splitter(selector, split_size=100, axis=0)
#selector = helpers.Splitter(selector, split_size=100, axis=0)
#selector = helpers.Validation(selector, test_fraction=0.25, stratified=True)
selector = helpers.Validation(selector, test_fraction=0.50, stratified=True)

speakers = range(0, len(speakers))

X_train = collections.defaultdict(list)
for input, target in selector.train:
    X_train[target].append(input.ravel())

models = []
for speaker in speakers:
    #clf = GMM(n_components=5) # TODO fix number of samples
    clf = GMM(n_components=3)
    clf.fit(X_train[speaker])
    models.append(clf)

errors = 0
observations = 0
for input, target in selector.test:
    #probs = np.array([model.predict_proba([input.ravel()])[0] for model in models])
    #probs = np.array([model.predict_proba([input.ravel()])[0] for model in models])
    probs = np.array([model.score([input.ravel()])[0] for model in models])

    #probs = []
    #for model in models:
    #    lpr = (mixture.log_multivariate_normal_density(np.array([input.ravel()]), model.means_, model.covars_, model.covariance_type) + np.log(model.weights_)) # probabilities of components
    #    #print(lpr)
    #    logprob = logsumexp(lpr, axis=1) # logsum to get probability of GMM
    #    prob = np.exp(logprob) # 0 < probs < 1

    #    probs.append(prob[0])
    #    #break

    #print(np.max(probs, axis=1))
    #y_hat = np.argmax(np.max(probs, axis=1))
    print(probs)
    print(np.exp(probs))
    y_hat = np.argmax(probs)
    errors += (y_hat != target)
    print('y_hat = %d == %d' % (y_hat, target))
    observations += 1


MSE = errors / observations
print('missrate = %.4f' % (MSE))
