
from __future__ import print_function

import numpy as np

import theano
import theano.tensor as T

import lasagne
import os
filepath = os.path.dirname(os.path.abspath(__file__))

class Network():

    MODEL_FOLDER = os.path.join(filepath, 'models/')

    def __init__(self, input_shape=None, output_size=None, input_var=None, target_var=None,
                 learning_rate=0.10, model_name='model-1'):
        self._input_shape   = input_shape
        self._output_size   = output_size
        self._input_var     = input_var
        self._target_var    = target_var
        self._learning_rate = learning_rate
        self._model_name    = model_name

    def build_net(self):
        print("Building model...")

        # Input layer, as usual:
        network = lasagne.layers.InputLayer(shape=self._input_shape, input_var=self._input_var)

        network = lasagne.layers.Conv2DLayer(incoming=network,
                    num_filters=16, filter_size=(8, 8),
                    nonlinearity=lasagne.nonlinearities.sigmoid)
                    #nonlinearity=lasagne.nonlinearities.rectify)

        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(4, 4))

        # A fully-connected layer of 32 units with 0 dropout on its inputs:
        network = lasagne.layers.DenseLayer(incoming=network, num_units=32,
                        nonlinearity=lasagne.nonlinearities.sigmoid)
                        #nonlinearity=lasagne.nonlinearities.rectify)

        network = lasagne.layers.DenseLayer(incoming=network, num_units=self._output_size,
                        nonlinearity=lasagne.nonlinearities.softmax)

        self._network = network

    def compile_model(self):
        print("Compiling functions...")

        # Create a loss expression for training
        prediction = lasagne.layers.get_output(self._network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, self._target_var)
        loss = loss.mean()

        # Create update expressions for training
        params = lasagne.layers.get_all_params(self._network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
                    loss, params, learning_rate=self._learning_rate, momentum=0.9)

        # Create a loss expression for validation/testing.
        test_prediction = lasagne.layers.get_output(self._network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, self._target_var)
        test_loss = test_loss.mean()

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([self._input_var, self._target_var], loss, updates=updates)

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([self._input_var, self._target_var], test_loss)

        # Create prediction function
        self._predict_fn = theano.function([self._input_var], test_prediction)

        self._train_fn = train_fn
        self._val_fn = val_fn

    def predict(self, X):
        return self._predict_fn(X)

    def save_model(self):
        np.savez(os.path.join(self.MODEL_FOLDER, '%s.npz' % (self._model_name)),
                *lasagne.layers.get_all_param_values(self._network))

    def load_model(self):
        with np.load(os.path.join(self.MODEL_FOLDER, '%s.npz' % (self._model_name))) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self._network, param_values)
