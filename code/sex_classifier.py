
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

def create_selector(usage):
    selector = timit.FileSelector(usage=usage, dialect='dr1')
    selector = helpers.Spectrogram(selector, nperseg=256, noverlap=128, normalize_signal=True)
    selector = helpers.Truncate(selector, truncate=300, axis=2)
    selector = helpers.Normalize(selector)
    selector = helpers.Minibatch(selector, cache=True)
    return selector

test_selector = create_selector('test')
train_selector = create_selector('train')

epochs = 300

# Setup data containers and matplotlib
train_loss_arr = np.zeros(epochs)
test_loss_arr = np.zeros(epochs)
epoch_arr = np.arange(1, epochs + 1)

fig, ax = plt.subplots()
train_points, = ax.plot(epoch_arr, train_loss_arr, label='train')
test_points, = ax.plot(epoch_arr, test_loss_arr, label='test')
plt.ylim(0, 1)
plt.legend()
plt.ion()

for test_data in test_selector:
    print(np.mean(cnn.predict(test_data[0]), axis=0))

# Train network
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

    train_points.set_data(epoch_arr, train_loss_arr)
    test_points.set_data(epoch_arr, test_loss_arr)
    plt.pause(0.1)

missclassifications = 0
observations = 0
for (test_input, test_target) in test_selector:
    predict = np.argmax(cnn.predict(test_input), axis=1)
    observations += len(predict)
    missclassifications += np.sum(predict != test_target)

print('missrate: %f' % (missclassifications / observations))

plt.ioff()
plt.show()
