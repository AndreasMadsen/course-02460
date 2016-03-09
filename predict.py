
from __future__ import print_function

import numpy as np

import theano
import theano.tensor as T

import lasagne

from network import Network
from helper_functions import iterate_minibatches
from load_data import DataModel, DataFilter


def main(learning_rate=0.001, batch_size=50):
    # Initialize data model
    data_filter = DataFilter(usages=['train'], dialects=['dr1'], speakers=None)
    data_model = DataModel(data_filter=data_filter)

    # Load data
    X, Y = data_model.data

    input_shape = (None, 1, X.shape[2], X.shape[3])
    output_size = Y.shape[1]

    # Prepare Theano variables for inputs and targets
    input_var = T.ftensor4('inputs')     # float32
    target_var = T.imatrix('targets')    # int32 (overkill)

    network = Network(input_shape=input_shape, output_size=output_size, input_var=input_var,
                      target_var=target_var, learning_rate=learning_rate)

    # Build network
    network.build_net()

    # Compile functions
    network.compile_model()

    # Load weights
    network.load_model()

    # Create prediction probabilities
    y_hat = network.predict(X)

    # Get max idx (predicted class)
    y_hat = np.argmax(y_hat, axis=1)
    Y     = np.argmax(Y,     axis=1)

    for i in range(0, Y.shape[0]):
        print((Y[i], y_hat[i]))

    MSE = np.mean((y_hat == Y).astype(int))
    print('Predicted %d observations' % (Y.shape[0]))
    print('MSE: %.2f' % (MSE))


if __name__ == '__main__':
    main()
