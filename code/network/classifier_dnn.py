
import lasagne
import theano.tensor as T
import numpy as np

import helpers
import early_stopping

from network.abstraction import NetworkAbstraction

class ClassifierDNN(NetworkAbstraction):
    def __init__(self, *args,
                 learning_rate=0.001, momentum=0.9, **kwargs):
        super().__init__(
            input_var=T.fmatrix('input'),
            target_var=T.ivector('target'),
            *args, **kwargs
        )

        self._learning_params = {
            'learning_rate': learning_rate,
            'momentum': momentum
        }

    def _build_network(self):
        network = lasagne.layers.InputLayer(
            shape=(None, ) + self.input_shape,
            input_var=self.input_var
        )

        network = lasagne.layers.DenseLayer(
            incoming=network,
            num_units=20,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform('relu')
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

    def fit(self, X, y, max_epochs=200):
        minibatch = helpers.Minibatch(zip(X, y))
        stoppage = early_stopping.PrecheltStopping()

        # Train network
        for epoch in range(max_epochs):
            train_loss = 0
            train_batches = 0

            for train_data in minibatch:
                train_loss += self.train(*train_data)
                train_batches += 1

            train_loss_current_epoch = train_loss / train_batches

            self._print("Epoch %d: Train Loss %g" % (epoch + 1, train_loss_current_epoch))

            if stoppage.is_converged(train_loss_current_epoch):
                self._print("Stopping early")
                break

    def score(self, X, y):
        predict = np.argmax(self.predict_proba(X), axis=1)
        return np.mean(predict == y)

    def predict_proba(self, X):
        return self.predict(X.astype('float32'))
