
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
    nn_si.add_regularizer(network.regularizer.ScaleInvariant(1e+1))
    nn_si.compile()

    nn_oi = network.ClassifierDNN(input_shape=(2, ), output_units=2)
    nn_oi.add_regularizer(network.regularizer.OffsetInvariant(1e+1))
    nn_oi.compile()

    names = ["NN", "NN + L2", "NN + Scale Invariant", "NN + Offset Invariant"]
    classifiers = [
        #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        #LinearDiscriminantAnalysis(),
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

#
# From http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
#
h = .02  # step size in the mesh
n_datasets = 4
n_classifiers = 4
samples = 100
levels = np.linspace(0, 1, 10)

figure = plt.figure(figsize=(12.5, 10))
i = 1
# iterate over datasets
for ds_i, (ds_name, ds) in enumerate(zip(*build_datasets(n_samples=samples))):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(n_datasets, n_classifiers + 1, i)
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_ylabel(ds_name, fontsize=16)
    i += 1

    # iterate over classifiers
    for name, clf in zip(*build_classifiers()):
        ax = plt.subplot(n_datasets, n_classifiers + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if (ds_i == 0): ax.set_title(name, fontsize=16)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=16, horizontalalignment='right')
        i += 1

figure.subplots_adjust(left=.02, right=.98)
plt.savefig('2d_classifier.pdf')
plt.show()
