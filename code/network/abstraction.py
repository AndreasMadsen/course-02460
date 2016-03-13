
import lasagne
import theano

class NetworkAbstraction:
    def __init__(self, input_shape, output_units, input_var, target_var, *args,
                 verbose=False, **kwargs):
        self.input_shape = input_shape
        self.output_units = output_units

        self.input_var = input_var
        self.target_var = target_var

        self._verbose = verbose
        self._compiled = False
        self._print('Network initalized')

    def _print(self, msg):
        if (self._verbose): print(type(self).__name__ + ': ' + msg)

    def _build_network(self):
        raise NotImplementedError

    def _loss_function(self):
        raise NotImplementedError

    def _update_function(self):
        raise NotImplementedError

    def compile(self):
        self._print('Compiling network ...')
        network = self._build_network()
        parameters = lasagne.layers.get_all_params(network, trainable=True)

        # Build train function
        prediction_train = lasagne.layers.get_output(network)
        loss_train = self._loss_function(prediction_train)
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
        loss_test = self._loss_function(prediction_test)

        # Compile loss function
        self._loss_fn = theano.function(
            [self.input_var, self.target_var],
            loss_test
        )
        self._print('Loss function compiled')

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
