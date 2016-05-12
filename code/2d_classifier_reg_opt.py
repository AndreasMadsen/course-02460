
import os
import sys
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification

import network

classifier_name = [
    "NN + L2",
    "NN + Scale Invariant",
    "NN + Offset Invariant"
]

dataset_name = [
    "moons",
    "circles",
    "linear",
    "xor"
]

def get_classifier(reg_param, classifier_idx=0):
    clf = None
    if classifier_idx == 0:
        nn_l2 = network.ClassifierDNN(input_shape=(2, ), output_units=2)
        nn_l2.add_regularizer(network.regularizer.WeightDecay(reg_param))
        nn_l2.compile()
        clf = nn_l2
    elif classifier_idx == 1:
        nn_si = network.ClassifierDNN(input_shape=(2, ), output_units=2)
        nn_si.add_regularizer(network.regularizer.ScaleInvariant(reg_param))
        nn_si.compile()
        clf = nn_si
    elif classifier_idx == 2:
        nn_oi = network.ClassifierDNN(input_shape=(2, ), output_units=2)
        nn_oi.add_regularizer(network.regularizer.OffsetInvariant(reg_param))
        nn_oi.compile()
        clf = nn_oi
    else:
        raise KeyError('Wrong classifier index received!')
    return classifier_name[classifier_idx], clf

def xor_scale_invariant(n_samples=100, noise=0.4):
    # Create true data
    X = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, 2))
    # Create labels
    y = np.zeros(n_samples, dtype='int32')
    y[(X[:, 0] * X[:, 1]) > 0] = 1
    # Offset true data
    X += np.random.uniform(low=-1.0, high=1.0, size=(n_samples, 1)) * noise
    return (X, y)

def get_dataset(dataset_idx=0, n_samples=100):
    ds = None
    if dataset_idx == 0:
        ds = make_moons(n_samples=n_samples, noise=0.3)
    elif dataset_idx == 1:
        return "circles", make_circles(n_samples=n_samples, noise=0.2, factor=0.5)
    elif dataset_idx == 2:
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
        X += 2 * np.random.uniform(size=X.shape)
        linearly_separable = (X, y)
        ds = linearly_separable
    elif dataset_idx == 3:
        ds = xor_scale_invariant(n_samples=n_samples)
    else:
        raise KeyError('Wrong dataset index received!')
    return dataset_name[dataset_idx], ds

def main(dataset_idx, classifier_idx, nfolds):
    print("Running Cross-validation for %s on the %s dataset.." % (
        classifier_name[classifier_idx],
        dataset_name[dataset_idx]
    ))

    # Setup
    samples = 100
    reg_values = [1] + [x* 10 ** (-y) for y in range(1, 11) for x in [5, 2, 1]] + [0]
    scores = np.zeros((nfolds, len(reg_values)))

    for k in range(0, nfolds):
        print('Fold %d/%d)' % (k+1, nfolds))
        # Generate dataset
        ds_name, ds = get_dataset(dataset_idx, samples)
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
        for i, reg_val in enumerate(reg_values):

            # Get classifier
            clf_name, clf = get_classifier(reg_val, classifier_idx)

            # Predict
            clf.fit(X_train, y_train)
            scores[k,i] = clf.score(X_test, y_test)
            print('paramater value: %e obtained score: %.4f' % (reg_val, scores[k,i]))

    # Save results
    print('Saving result..')
    reg_values = np.array(reg_values)
    X = np.vstack((reg_values, scores))
    filename = 'cv-scores-ds-%d-clf-%d' % (dataset_idx, classifier_idx)
    np.savetxt('./output/cv_results/%s.csv' % (filename), X)

if __name__ == '__main__':
    kwargs = {
        'dataset_idx': 0,
        'classifier_idx': 0,
        'nfolds': 50
    }
    #if len(sys.argv) > 1:
    #    kwargs['dataset_idx'] = int(sys.argv[1])
    #if len(sys.argv) > 2:
    #    kwargs['classifier_idx'] = int(sys.argv[2])

    if 'DATASET_IDX' in os.environ.keys():
        kwargs['dataset_idx'] = int(os.environ['DATASET_IDX'])
    if 'CLASSIFIER_IDX' in os.environ.keys():
        kwargs['classifier_idx'] = int(os.environ['CLASSIFIER_IDX'])
    if 'NFOLDS' in os.environ.keys():
        kwargs['nfolds'] = int(os.environ['NFOLDS'])

    main(**kwargs)
