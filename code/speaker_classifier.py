
import os
import numpy as np
import lasagne
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

import timit
import network
import helpers


display_active = "DISPLAY" in os.environ and len(os.environ["DISPLAY"]) > 0

# Create data selector object
selector = timit.FileSelector(dialect=None)
selector = helpers.TargetType(selector, target_type='speaker')
speakers = selector.labels
selector = helpers.Spectrogram(selector, nperseg=256, noverlap=128, normalize_signal=True)
selector = helpers.Truncate(selector, truncate=300, axis=2)
selector = helpers.Normalize(selector)
selector = helpers.Filter(selector, min_count=10)
selector = helpers.Validation(selector, test_fraction=0.25, stratified=True)

train_selector = helpers.Minibatch(selector.train)
test_selector  = helpers.Minibatch(selector.test)

cnn = network.DielemanCNN(input_shape=(1, 129, 300), output_units=len(speakers.keys()),
                          verbose=True, learning_rate=0.001,
                          regularization=True, dropout=True)
cnn.compile()

epochs = 100

# Setup data containers and matplotlib
train_loss_arr = np.zeros(epochs)
test_loss_arr = np.zeros(epochs)
epoch_arr = np.arange(1, epochs + 1)

if display_active:
    fig, ax = plt.subplots()
    train_points, = ax.plot(epoch_arr, train_loss_arr, label='train')
    test_points, = ax.plot(epoch_arr, test_loss_arr, label='test')
    plt.ylim(0, 1)
    plt.legend()
    plt.ion()

# Train network
max_error = 1.0
for epoch in range(epochs):
    train_loss = 0
    train_batches = 0

    test_loss = 0
    test_batches = 0

    for train_data in train_selector:
        train_loss += cnn.train(*train_data)
        train_batches += 1

    for test_data in test_selector:
        test_loss += cnn.loss(*test_data)
        test_batches += 1

    print("Epoch %d: Train Loss %g, Test Loss %g" % (
          epoch + 1, train_loss / train_batches, test_loss / test_batches))

    train_loss_arr[epoch] = train_loss / train_batches
    test_loss_arr[epoch] = test_loss / test_batches

    if display_active:
        train_points.set_data(epoch_arr, train_loss_arr)
        test_points.set_data(epoch_arr, test_loss_arr)
        max_error = max(max_error, np.max(np.concatenate((train_loss_arr.ravel(), test_loss_arr.ravel()))))
        ax.set_ylim([0, max_error])
        plt.pause(0.1)

missclassifications = 0
observations = 0
for (test_input, test_target) in test_selector:
    predict = np.argmax(cnn.predict(test_input), axis=1)
    observations += len(predict)
    missclassifications += np.sum(predict != test_target)

print('missrate: %f' % (missclassifications / observations))

if display_active:
    plt.ioff()
    plt.show()
