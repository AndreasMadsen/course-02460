
import lasagne
import theano.tensor as T

from network.abstraction import NetworkAbstraction

class Logistic(NetworkAbstraction):
    def __init__(self, *args,
                 learning_rate=0.001, momentum=0.9, **kwargs):
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

    def _build_network(self):
        network = lasagne.layers.InputLayer(
            shape=(None, ) + self.input_shape[0:2],
            input_var=self.input_var.mean(axis=3)
        )

        network = lasagne.layers.DenseLayer(
            incoming=network,
            num_units=self.output_units,
            nonlinearity=lasagne.nonlinearities.softmax
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
