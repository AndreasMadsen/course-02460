
from __future__ import print_function

import sys
import os
import numpy as np
import time

from load_data import DataModel
from helper_functions import iterate_minibatches

import theano
import theano.tensor as T

import lasagne

def build_cnn(input_shape, output_size, filter_height, input_var=None):
    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    # Convolutional layer with 16 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
                network, num_filters=16, filter_size=(8, 8),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())

    print('layers output size:')
    print(lasagne.layers.get_output_shape(network))

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(4, 4))

    print(lasagne.layers.get_output_shape(network))

    # Another convolution with 16 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
                network, num_filters=16, filter_size=(8, 8),
                nonlinearity=lasagne.nonlinearities.rectify)
    print(lasagne.layers.get_output_shape(network))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(4, 4))
    print(lasagne.layers.get_output_shape(network))

    # A fully-connected layer of 64 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(network, p=.5),
                num_units=64,
                nonlinearity=lasagne.nonlinearities.rectify)
    print(lasagne.layers.get_output_shape(network))

    # A fully-connected layer of 32 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(network, p=.5),
                num_units=32,
                nonlinearity=lasagne.nonlinearities.rectify)
    print(lasagne.layers.get_output_shape(network))

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(network, p=.5),
                num_units=output_size,
                nonlinearity=lasagne.nonlinearities.softmax)
    print(lasagne.layers.get_output_shape(network))

    return network

def main(loss_function=lasagne.objectives.categorical_crossentropy, num_epochs=100,
         learning_rate=0.01, batch_size=50):

    # Initialize data model
    data_model = DataModel()

    # Load data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = data_model.data

    #X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
    #X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1], X_val.shape[2]))
    #X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))

    input_shape = (None, 1, X_train.shape[2], X_train.shape[3])
    output_size = Y_test.shape[1]

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')     # float64
    target_var = T.imatrix('targets')   # int32 (overkill)

    print("Building model and compiling functions...")
    network = build_cnn(input_shape=input_shape, output_size=output_size,
                        filter_height=X_train.shape[2], input_var=input_var)

    # Create a loss expression for training
    prediction = lasagne.layers.get_output(network)
    loss = loss_function(prediction, target_var)
    loss = loss.mean()

    # Create update expressions for training
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate=learning_rate, momentum=0.9)

    # Create a loss expression for validation/testing.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = loss_function(test_prediction, target_var)
    test_loss = test_loss.mean()

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], test_loss)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, Y_train, batch_size, shuffle=True):
            inputs, targets = batch
            print(train_err)
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, Y_val, batch_size, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, Y_test, batch_size, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))


if __name__ == '__main__':
    main()
