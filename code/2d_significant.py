
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import network

"""
with open('./output/classifier_2d_opt_param_values.json') as fd:
    best_regualizer_parameters = json.load(fd)
"""
def build_classifiers(model):
    """
    regs = best_regualizer_parameters[model]

    nn_simple = network.ClassifierDNN(input_shape=(2, ), output_units=2)
    nn_simple.compile()

    nn_l2 = network.ClassifierDNN(input_shape=(2, ), output_units=2)
    nn_l2.add_regularizer(network.regularizer.WeightDecay(regs['NN + L2']))
    nn_l2.compile()

    nn_si = network.ClassifierDNN(input_shape=(2, ), output_units=2)
    nn_si.add_regularizer(network.regularizer.ScaleInvariant(regs['NN + Scale Invariant'], use_Rop=True))
    nn_si.compile()

    nn_oi = network.ClassifierDNN(input_shape=(2, ), output_units=2)
    nn_oi.add_regularizer(network.regularizer.OffsetInvariant(regs['NN + Offset Invariant'], use_Rop=True))
    
    nn_oi.compile()
    
    """
    nn_simple = network.ClassifierDNN(input_shape=(2, ), output_units=2, verbose=False)
    nn_simple.compile()

    nn_l2 = network.ClassifierDNN(input_shape=(2, ), output_units=2, verbose=False)
    nn_l2.add_regularizer(network.regularizer.WeightDecay(1e-1))
    nn_l2.compile()

    nn_si = network.ClassifierDNN(input_shape=(2, ), output_units=2, verbose=False)
    nn_si.add_regularizer(network.regularizer.ScaleInvariant(1e+1, use_Rop=True))
    nn_si.compile()

    nn_oi = network.ClassifierDNN(input_shape=(2, ), output_units=2, verbose=False)
    nn_oi.add_regularizer(network.regularizer.OffsetInvariant(3e+1, use_Rop=True))

    nn_oi.compile()
    
    nn_ri = network.ClassifierDNN(input_shape=(2, ), output_units=2)
    nn_ri.add_regularizer(network.regularizer.RotationInvariant(3e+0, use_Rop=True))
    nn_ri.compile()
    

    names = ["Random Forest", "Linear Discriminant Analysis", "NN", "NN + L2", "NN + Scale Invariant", "NN + Offset Invariant","NN + Rotation Invariant"]
    classifiers = [
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        LinearDiscriminantAnalysis(),
        nn_simple, nn_l2, nn_si, nn_oi, nn_ri
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

def rotation_matrix(angle):
    return np.matrix([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])

def ray_bow(n_samples=100, noise=0.15):
    #constructs each group as a cross and then rotates one group
    r_center = np.random.uniform(low=-1, high=1, size=n_samples/2)
    outskirts = np.hstack((np.random.uniform(low=0.2, high=1, size=n_samples/4),
            -np.random.uniform(low=0.4, high=1, size=n_samples/4)))
    np.random.shuffle(outskirts)

    group1_y = r_center
    group2_x = outskirts
    


    group1_x = [np.random.uniform(low=-noise, high=noise) for y in group1_y]
    group2_y = [-np.random.uniform(low=-noise, high=noise)*y for y in group2_x]

    y = np.hstack((np.zeros(n_samples/2,dtype='int32'), np.ones(n_samples/2)))
    X = np.vstack((
            np.hstack((group1_x, group1_y)),
            np.hstack((group2_x, group2_y))
            )).T

    # rotate 1 class off axis
    rotation_45 = rotation_matrix(np.pi/4)
    X[:n_samples/2,] = np.dot(X[:n_samples/2,], rotation_45)

    # rotate half of each class to construct cross
    rotation_90 = rotation_matrix(np.pi/2)
    X[:n_samples/4,] = np.dot(X[:n_samples/4,], rotation_90)
    X[n_samples/2:n_samples*3/4,] = np.dot(X[n_samples/2:n_samples*3/4,], rotation_90)
    
    #rotate cross 60
    #rotation_60 = rotation_matrix(np.pi/3)
    #X[n_samples/2:n_samples*5/8,] = np.dot(X[n_samples/2:n_samples*5/8,], rotation_60)

    return (X, y)

def circle_of_life(n_samples=100, seperation=-0.1, n_rings=2):
    interval_dist=1/n_rings

    angles = np.random.uniform(low=0, high=2*np.pi, size=n_samples)



    floor_inner = np.round(n_samples/n_rings)
    inner_circle = np.random.uniform(low=0, high=interval_dist, size=floor_inner)
    
    floor_mid = n_samples - floor_inner #np.round(n_samples/4)
    mid_circle = np.random.uniform(low=interval_dist+seperation, high=2*interval_dist+seperation, size=floor_mid)

    #floor_outer = np.round(n_samples/4)
    #outer_circle = np.random.uniform(low=2*interval_dist+2*seperation, high=3*interval_dist+2*seperation, size=floor_outer)

    #floor_last = n_samples - floor_outer - floor_mid - floor_inner
    #last_circle = np.random.uniform(low=3*interval_dist+3*seperation, high=4*interval_dist+3*seperation, size=floor_last)


    distances = np.hstack((inner_circle, mid_circle))#, outer_circle,last_circle))

    X = np.vstack((distances*np.cos(angles),distances*np.sin(angles))).T
    y = np.array([1]*floor_inner+[0]*floor_mid)#+[1]*floor_outer+[0]*floor_last)

    return (X, y)

def make_noise(input, noise=0.0):
    # take noise-perencet of x,y indicies and set them to random classes/locations
    X = input[0]
    y = input[1]
    n = X.shape[0]
    indicies = np.random.choice(range(n), size=np.round(n*noise), replace=False)

    x_rand = np.random.uniform(low=-1, high=1, size=(n*noise,2))
    first_half = np.round(n*noise/2)
    second_half = n*noise - first_half
    y_rand = np.array([0]* first_half + [1]*second_half)
    X[indicies, :] = x_rand
    y[indicies] = y_rand
    return (X, y)

def gandalf(n_samples=100):
    group_n = int(np.round(n_samples/4))
    cov = np.matrix([[1, 0.8], [0.8,1]])/100
    X_1 = np.random.multivariate_normal([0.0,0.0], 4*cov, size=2*group_n)
    y_1 = [0]*group_n*2

    X_2 = np.random.multivariate_normal([-0.5,0.5], cov, size=group_n)
    y_2 = [1]*group_n

    X_3 = np.random.multivariate_normal([0.5,-0.5], cov, size=group_n)
    y_3 = [1]*group_n

    X = np.concatenate((X_1, X_2, X_3), axis=0)
    y = np.array(y_1 + y_2 + y_3)
    return (X, y)    



def build_datasets(n_samples=100):
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
    X += 2 * np.random.uniform(size=X.shape)
    linearly_separable = (X, y)

    names = ['moons', 'circles', 'linear', 'xor', 'diagonal','rays','dist']
    datasets = [ make_moons(n_samples=n_samples, noise=0.3),
                 make_circles(n_samples=n_samples, noise=0.2, factor=0.5),
                 linearly_separable,
                 xor_scale_invariant(n_samples=n_samples),
                make_noise(gandalf(n_samples=n_samples), noise=0.3),
                make_noise(ray_bow(n_samples=n_samples), noise=0.05), 
                make_noise(circle_of_life(n_samples=n_samples), noise=0)]
    return (names, datasets)



samples = 400
n_trials = 25

with open('./output/classifer-significance-'+str(n_trials)+'-trials.csv', 'w') as fd:
    writer = csv.DictWriter(fd,
                            ['trial', 'dataset', 'model', 'missclassification'],
                            dialect='unix')
    writer.writeheader()

    for trial in range(0, n_trials):
        # iterate over datasets
        for ds_name, ds in zip(*build_datasets(n_samples=samples)):
            # preprocess dataset, split into training and test part
            X, y = ds
            X = StandardScaler().fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

            # iterate over classifiers
            for clf_name, clf in zip(*build_classifiers(ds_name)):
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)

                print('[%d] - %s using %s: %f' % (trial, ds_name, clf_name, 1 - score))

                writer.writerow({
                    'trial': trial,
                    'dataset': ds_name,
                    'model': clf_name,
                    'missclassification': 1 - score
                })
