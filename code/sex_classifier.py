
import timit

import numpy as np
import lasagne
import theano
import theano.tensor as T
import scipy.signal
import scipy.io.wavfile

# create Theano variables for input and target minibatch
input_var = T.tensor4('X')
target_var = T.ivector('y')

# create a small convolutional neural network
network = lasagne.layers.InputLayer(shape=(None, 1, 129, 250), input_var=input_var)

network = lasagne.layers.Conv2DLayer(network, num_filters=16, filter_size=(16, 16),
                                     nonlinearity=lasagne.nonlinearities.leaky_rectify)

network = lasagne.layers.Conv2DLayer(network, num_filters=2, filter_size=(3, 3),
                                     nonlinearity=lasagne.nonlinearities.leaky_rectify)

network = lasagne.layers.Pool2DLayer(network, (3, 3), stride=2, mode='max')

network = lasagne.layers.DenseLayer(network, num_units=32,
                                    nonlinearity=lasagne.nonlinearities.leaky_rectify,
                                    W=lasagne.init.Orthogonal())

network = lasagne.layers.DenseLayer(network, num_units=2,
                                    nonlinearity=lasagne.nonlinearities.softmax,
                                    W=lasagne.init.Orthogonal())

# create loss function
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

# create parameter update expressions
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params,
                                            learning_rate=0.01, momentum=0.9)

# compile training function that updates parameters and returns training loss
train_fn = theano.function([input_var, target_var], loss, updates=updates)

# train network (assuming you've got some training data in numpy arrays)
print("training network")
train_selector = timit.FileSelector(usage='train', dialect='dr1', texttype='SI')
for epoch in range(100):
    loss = 0
    items = 0
    for item in train_selector:

        # Get spectogram
        spectogram = item.spectogram()
        if (spectogram.shape[1] < 250): continue
        spectogram = spectogram[:, 0:250]
        spectogram = spectogram.reshape(1, 1, *spectogram.shape)
        items += 1

        target = np.asarray([int(item.sex == 'f')], dtype='int32')

        loss += train_fn(spectogram, target)
        print(loss)
    print("Epoch %d: Loss %g" % (epoch + 1, loss / items))

# use trained network for predictions
test_prediction = lasagne.layers.get_output(network, deterministic=True)
predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))

# Print predictions
print("predicting network")
test_selector = timit.FileSelector(usage='test', dialect='dr1', texttype='SI')
for item in test_selector:

    # Get spectogram
    spectogram = spectogram()
    if (spectogram.shape[1] < 250): continue
    spectogram = spectogram[:, 0:250]
    spectogram = spectogram.reshape(1, 1, *spectogram.shape)

    print("(%s, %s): %d (actual: %s)" % (
        item.speaker, item.sentense,
        ['m', 'f'][predict_fn(spectogram)], item.sex
    ))
