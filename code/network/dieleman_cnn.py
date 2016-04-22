
import lasagne
import theano.tensor as T

from network.abstraction import NetworkAbstraction

class DielemanCNN(NetworkAbstraction):
    def __init__(self, *args,
                 learning_rate=0.001, momentum=0.9, dropout=False, **kwargs):
        super().__init__(
            input_var=T.ftensor4('input'),
            target_var=T.ivector('target'),
            *args, **kwargs
        )

        if (len(self.input_shape) != 3):
            raise ValueError('expected input shape with 3 elements')

        self._learning_params = {
            'learning_rate': learning_rate,
            'momentum': momentum
        }

        self._dropout = dropout

    def _build_network(self):
        network = lasagne.layers.InputLayer(
            shape=(None, ) + self.input_shape,
            input_var=self.input_var
        )

        network = lasagne.layers.Conv2DLayer(
            incoming=network,
            num_filters=32, filter_size=(self.input_shape[1], 8),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform('relu')
        )

        network = lasagne.layers.MaxPool2DLayer(
            incoming=network,
            pool_size=(1, 4)
        )

        network = lasagne.layers.Conv2DLayer(
            incoming=network,
            num_filters=32, filter_size=(1, 8),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform('relu')
        )

        network = lasagne.layers.MaxPool2DLayer(
            incoming=network,
            pool_size=(1, 4)
        )

        if (self._dropout):
            network = lasagne.layers.DropoutLayer(network, p=0.5)

        network = lasagne.layers.DenseLayer(
            incoming=network,
            num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform('relu')
        )

        network = lasagne.layers.DenseLayer(
            incoming=network,
            num_units=self.output_units,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.GlorotUniform(1.0)
        )

        return network

    def _loss_function(self, prediction):
        loss = lasagne.objectives.categorical_crossentropy(prediction, self.target_var)
        return loss.mean()

    def _update_function(self, loss, parameters):
        update = lasagne.updates.nesterov_momentum(
            loss, parameters,
            **self._learning_params
        )
        return update
