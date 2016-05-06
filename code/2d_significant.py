
import csv

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import network

def build_classifiers():
    nn_simple = network.ClassifierDNN(input_shape=(2, ), output_units=2)
    nn_simple.compile()

    nn_l2 = network.ClassifierDNN(input_shape=(2, ), output_units=2)
    nn_l2.add_regularizer(network.regularizer.WeightDecay(1e-1))
    nn_l2.compile()

    nn_si = network.ClassifierDNN(input_shape=(2, ), output_units=2)
    nn_si.add_regularizer(network.regularizer.ScaleInvariant(1e-1, use_Rop=True))
    nn_si.compile()

    nn_oi = network.ClassifierDNN(input_shape=(2, ), output_units=2)
    nn_oi.add_regularizer(network.regularizer.OffsetInvariant(1e-1, use_Rop=True))
    nn_oi.compile()

    names = ["Random Forest", "Linear Discriminant Analysis", "NN", "NN + L2", "NN + Scale Invariant", "NN + Offset Invariant"]
    classifiers = [
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        LinearDiscriminantAnalysis(),
        nn_simple, nn_l2, nn_si, nn_oi
    ]
    return (names, classifiers)

def xor_scale_invariant(n_samples=100, noise=0.4):
    # Create true data
    X = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, 2))
    # Create labels
    y = np.zeros(n_samples, dtype='int32')
    y[(X[:, 0] * X[:, 1]) > 0] = 1
    # Offset true data
    X += np.random.uniform(low=-1.0, high=1.0, size=(n_samples, 1)) * noise
    return (X, y)

def build_datasets(n_samples=100):
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
    X += 2 * np.random.uniform(size=X.shape)
    linearly_separable = (X, y)

    names = ['moons', 'circles', 'linear', 'xor']
    datasets = [make_moons(n_samples=n_samples, noise=0.3),
                make_circles(n_samples=n_samples, noise=0.2, factor=0.5),
                linearly_separable,
                xor_scale_invariant(n_samples=n_samples)]
    return (names, datasets)

samples = 25

with open('./output/classifer-significance-25-samples.csv', 'w') as fd:
    writer = csv.DictWriter(fd,
                            ['trial', 'dataset', 'model', 'score'],
                            dialect='unix')
    writer.writeheader()

    for trial in range(0, 100):
        # iterate over datasets
        for ds_name, ds in zip(*build_datasets(n_samples=samples)):
            # preprocess dataset, split into training and test part
            X, y = ds
            X = StandardScaler().fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

            # iterate over classifiers
            for clf_name, clf in zip(*build_classifiers()):
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)

                print('[%d] - %s using %s: %f' % (trial, ds_name, clf_name, score))

                writer.writerow({
                    'trial': trial,
                    'dataset': ds_name,
                    'model': clf_name,
                    'score': score
                })
