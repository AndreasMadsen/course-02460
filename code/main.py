
from __future__ import print_function

import sys
import os
import numpy as np
import time
filepath = os.path.dirname(os.path.abspath(__file__))

from load_data import DataModel, DataFilter
from helper_functions import iterate_minibatches

import theano
import theano.tensor as T

import lasagne

from network import Network

def main(num_epochs=100, learning_rate=0.001, batch_size=50):

    # Initialize data model
    data_filter = DataFilter(usages=['train'], dialects=['dr1'], speakers=None)
    data_model = DataModel(data_filter=data_filter)

    # Load data
    #X_train, Y_train, X_val, Y_val, X_test, Y_test = data_model.data
    X, Y = data_model.data


    #X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
    #X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1], X_val.shape[2]))
    #X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))

    input_shape = (None, 1, X.shape[2], X.shape[3])
    output_size = Y.shape[1]

    # Prepare Theano variables for inputs and targets
    input_var = T.ftensor4('inputs')     # float32
    #target_var = T.imatrix('targets')   # int32 (overkill)
    target_var = T.imatrix('targets')

    network = Network(input_shape=input_shape, output_size=output_size, input_var=input_var,
                      target_var=target_var, learning_rate=learning_rate)

    # Build network
    network.build_net(filter_height=X.shape[2])

    # Compile functions
    network.compile_model()

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    train_errs = np.zeros((num_epochs))
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X, Y, batch_size, shuffle=True):
            inputs, targets = batch
            train_errs[epoch] += network._train_fn(inputs, targets)
            #(train_errs[epoch])
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_errs[epoch] / train_batches))

    # After training, we compute and print the test error:
    test_err = 0
    test_batches = 0
    for batch in iterate_minibatches(X, Y, batch_size, shuffle=False):
        inputs, targets = batch
        err = network._val_fn(inputs, targets)
        test_err += err
        test_batches += 1
    print("Final results:")
    print("  training loss:\t\t\t{:.6f}".format(test_err / test_batches))

    # Save training errors to file
    np.savetxt(os.path.join(filepath, '../errors/%s.csv' % ('train_err')), train_errs, delimiter=",")

    # Save model weights
    network.save_model()



if __name__ == '__main__':
    main()
