
import numpy as np
import lasagne
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

import timit
import network
import helpers

cnn = network.SimpleCNN(input_shape=(1, 129, 300), output_units=2, verbose=True)
cnn.compile()

# File selectors
test_selector = helpers.MinibatchSpectogram(
    timit.FileSelector(usage='test', dialect='dr1'),
    nfft=256, truncate_time=300, noverlap=128)

train_selector = helpers.MinibatchSpectogram(
    timit.FileSelector(usage='train', dialect='dr1'),
    nfft=256, truncate_time=300, noverlap=128)

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
    train_batches = 0

    test_loss = 0
    test_batches = 0

    for train_data in train_selector:
        train_loss += cnn.train(*train_data)
        train_batches += 1

    for test_data in test_selector:
        test_loss += cnn.loss(*train_data)
        test_batches += 1

    print("Epoch %d: Train Loss %g, Test Loss %g" % (
          epoch + 1, train_loss / train_batches, test_loss / test_batches))

    train_loss_arr[epoch] = train_loss / train_batches
    test_loss_arr[epoch] = test_loss / test_batches

    train_points.set_data(epoch_arr, train_loss_arr)
    test_points.set_data(epoch_arr, test_loss_arr)
    plt.pause(0.1)

plt.ioff()
plt.show()
