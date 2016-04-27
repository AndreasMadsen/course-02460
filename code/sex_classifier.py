
import os
import numpy as np
import lasagne
import theano
import theano.tensor as T

import timit
import network
import helpers
import plot
import early_stopping

def create_selector(usage):
    selector = timit.FileSelector(usage=usage)#,dialect='dr1')
    selector = helpers.TargetType(selector, target='sex')
    selector = helpers.Spectrogram(selector, nperseg=256, noverlap=128, normalize_signal=True)
    selector = helpers.Truncate(selector, truncate=300, axis=2)
    selector = helpers.Normalize(selector)
    selector = helpers.Minibatch(selector)
    return selector

test_selector = create_selector('test')
train_selector = create_selector('train')

# cnn = network.DielemanCNN(input_shape=(1, 129, 300), output_units=2,
#                           verbose=True)
# cnn.add_regularizer(network.regularizer.WeightDecay(1e-1))

# cnn = network.Logistic(input_shape=(1, 129, 300), output_units=2,
#                        verbose=True, learning_rate=0.01)

# cnn = network.SimpleCNN(input_shape=(1, 129, 300), output_units=2, verbose=True)

cnn = network.DielemanCNN(input_shape=(1, 129, 300), output_units=2, verbose=True)
cnn.add_regularizer(network.regularizer.ScaleInvariant(1e-1))
cnn.compile()

epochs = 200

stoppage = early_stopping.PrecheltStopping(verbose=True)#verbose=True, alpha=1.5,interval_length=10

loss_plot = plot.LiveLoss(epochs)

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

    train_loss_current_epoch = train_loss / train_batches
    test_loss_current_epoch = test_loss  / test_batches

    print("Epoch %d: Train Loss %g, Test Loss %g" % (
          epoch + 1, train_loss_current_epoch,
          test_loss_current_epoch))

    loss_plot.set_loss(epoch,
                       train_loss_current_epoch,
                       test_loss_current_epoch)

    if stoppage.is_converged(test_loss_current_epoch):
        print("Stopping early")
        break

missclassifications = 0
observations = 0
for (test_input, test_target) in test_selector:
    predict = np.argmax(cnn.predict(test_input), axis=1)
    observations += len(predict)
    missclassifications += np.sum(predict != test_target)

print('missrate: %f' % (missclassifications / observations))

loss_plot.finish()
