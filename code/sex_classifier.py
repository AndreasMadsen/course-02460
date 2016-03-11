
import numpy as np
import lasagne
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

import timit
from helpers.minibatch_spectogram import MinibatchSpectogram

# create Theano variables for input and target minibatch
input_var = T.tensor4('X')
target_var = T.ivector('y')

# create a small convolutional neural network
network = lasagne.layers.InputLayer(shape=(None, 1, 257, 150), input_var=input_var)

network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(257, 8),
                                     nonlinearity=lasagne.nonlinearities.sigmoid)

network = lasagne.layers.MaxPool2DLayer(network, (1, 4))

network = lasagne.layers.DenseLayer(network, num_units=32,
                                    nonlinearity=lasagne.nonlinearities.sigmoid)

network = lasagne.layers.DenseLayer(network, num_units=2,
                                    nonlinearity=lasagne.nonlinearities.softmax)

# create loss function
train_prediction = lasagne.layers.get_output(network)
train_loss = lasagne.objectives.categorical_crossentropy(train_prediction, target_var)
train_loss = train_loss.mean()

# use trained network for predictions
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()

# create parameter update expressions
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(train_loss, params,
                                            learning_rate=0.001, momentum=0.9)

# compile functions
train_fn = theano.function([input_var, target_var], train_loss, updates=updates)
predict_fn = theano.function([input_var], test_prediction)
loss_fn = theano.function([input_var, target_var], test_loss)

# File selectors
test_selector = MinibatchSpectogram(
    timit.FileSelector(usage='test', dialect='dr1'))

train_selector = MinibatchSpectogram(
    timit.FileSelector(usage='train', dialect='dr1'))

# train network (assuming you've got some training data in numpy arrays)
print("training network")

train_loss_arr = np.zeros(100)
test_loss_arr = np.zeros(100)
epoch_arr = np.arange(1, 100 + 1)

fig, ax = plt.subplots()
train_points, = ax.plot(epoch_arr, train_loss_arr, label='train')
test_points, = ax.plot(epoch_arr, test_loss_arr, label='test')
plt.ylim(0, 1)
plt.legend()
plt.ion()

for epoch in range(100):
    train_loss = 0
    test_loss = 0
    items = 0

    for (train_data, test_data) in zip(train_selector, test_selector):
        train_batch_loss = train_fn(*train_data)
        test_batch_loss = loss_fn(*test_data)
        print(train_batch_loss, test_batch_loss)

        items += 1
        train_loss += train_batch_loss
        test_loss += test_batch_loss

    print("Epoch %d: Train Loss %g, Test Loss %g" % (
          epoch + 1, train_loss / items, test_loss / items))

    train_loss_arr[epoch] = train_loss / items
    test_loss_arr[epoch] = test_loss / items

    train_points.set_data(epoch_arr, train_loss_arr)
    test_points.set_data(epoch_arr, test_loss_arr)
    plt.pause(0.1)
