
import lasagne
import theano

class NetworkAbstraction:
    def __init__(self, input_shape, output_units, input_var, target_var,
                 regularization=0, dropout=False, *args, verbose=False, **kwargs):
        self.input_shape = input_shape
        self.output_units = output_units

        self.input_var = input_var
        self.target_var = target_var

        self._verbose = verbose
        self._compiled = False

        self._regularization = regularization
        self._dropout = dropout

        self._print('Network initalized')

    def _print(self, msg):
        if (self._verbose): print(type(self).__name__ + ': ' + msg)

    def _build_network(self):
        raise NotImplementedError

    def _loss_function(self):
        raise NotImplementedError

    def _update_function(self):
        raise NotImplementedError

    def compile(self, all_layers=False):
        self._print('Compiling network ...')
        network = self._build_network()
        parameters = lasagne.layers.get_all_params(network, trainable=True)

        # Build train function
        prediction_train = lasagne.layers.get_output(network)
        loss_train = self._loss_function(prediction_train, network)
        update = self._update_function(loss_train, parameters)

        # Compile train function
        self._train_fn = theano.function(
            [self.input_var, self.target_var],
            loss_train, updates=update
        )
        self._print('Train function compiled')

        # Build predict function
        prediction_test = lasagne.layers.get_output(network, deterministic=True)

        # Compile predict function
        self._predict_fn = theano.function([self.input_var], prediction_test)
        self._print('Predict function compiled')

        # Build loss function
        loss_test = self._loss_function(prediction_test, network)

        # Compile loss function
        self._loss_fn = theano.function(
            [self.input_var, self.target_var],
            loss_test
        )
        self._print('Loss function compiled')

        #
        if (all_layers):
            layer_outputs = []
            current_layer = network
            while (not isinstance(current_layer, lasagne.layers.InputLayer)):
                layer_outputs.append(
                    lasagne.layers.get_output(current_layer, deterministic=True)
                )
                current_layer = current_layer.input_layer
            layer_outputs = list(reversed(layer_outputs))

            self._all_layers_fn = theano.function(
                [self.input_var],
                [self.input_var] + layer_outputs
            )
            self._print('All layer output function compiled')

        # All function are now compiled
        self._compiled = True

    def train(self, input, target):
        if (not self._compiled): raise Exception('network is not compiled')
        return self._train_fn(input, target)

    def predict(self, input):
        if (not self._compiled): raise Exception('network is not compiled')
        return self._predict_fn(input)

    def loss(self, input, target):
        if (not self._compiled): raise Exception('network is not compiled')
        return self._loss_fn(input, target)

    def print_network_shapes(self):
        if (not self._compiled): raise Exception('network is not compiled')
        layers = lasagne.layers.get_all_layers(self._network)
        layers = list(filter(lambda x: not isinstance(x, lasagne.layers.DropoutLayer), layers))
        print('Network size:')
        for i, layer in enumerate(layers):
            shape = lasagne.layers.get_output_shape(layer)
            print('%d) %16s %s' % (i+1, type(layer).__name__, shape))

    def all_layer_outputs(self, input):
        if (not self._compiled): raise Exception('network is not compiled')
        if (not hasattr(self, '_all_layers_fn')):
            raise Exception('network is not compiled with all_layers=True')
        return self._all_layers_fn(input)
