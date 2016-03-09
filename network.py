
import theano
import theano.tensor as T

import lasagne

class Network():

    def __init__(self, input_shape=None, output_size=None, input_var=None, target_var=None,
                 learning_rate=0.10):
        self._input_shape   = input_shape
        self._output_size   = output_size
        self._input_var     = input_var
        self._target_var    = target_var
        self._learning_rate = learning_rate

    def build_net(self):
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
        print("Building model and compiling functions...")

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

        self._train_fn = train_fn
        self._val_fn = val_fn
