
import os
import numpy as np
import lasagne
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

import elsdsr
import network
import helpers
import plot
import early_stopping

# Create data selector object
selector = elsdsr.FileSelector(usage=None, sex=None, speaker=None)
#selector = helpers.Filter(selector, target='speaker', min_count=10, min_size=300, nperseg=256, noverlap=128)
selector = helpers.TargetType(selector, target='speaker')
speakers = selector.labels
selector = helpers.Spectrogram(selector, nperseg=256, noverlap=128, normalize_signal=True)
selector = helpers.Splitter(selector, split_size=100, axis=2)
#selector = helpers.Splitter(selector, split_size=300, axis=2)
#selector = helpers.Truncate(selector, truncate=300, axis=2)
selector = helpers.Normalize(selector)
selector = helpers.Validation(selector, test_fraction=0.25, stratified=True)

train_selector = helpers.Minibatch(selector.train)
test_selector  = helpers.Minibatch(selector.test)

#cnn = network.Logistic(input_shape=(1, 129, 100), output_units=len(speakers),
#                       verbose=True, learning_rate=0.1)
cnn = network.DielemanCNN(input_shape=(1, 129, 100), output_units=len(speakers),
                          verbose=True, learning_rate=0.001, dropout=False)
#cnn.add_regularizer(network.regularizer.WeightDecay(1e-2))
cnn.add_regularizer(network.regularizer.ScaleInvariant(1e-1))
#cnn.add_regularizer(network.regularizer.OffsetInvariant(1e+1, use_Rop=True))
cnn.compile()

epochs = 500
loss_plot = plot.LiveLoss(epochs)

stoppage = early_stopping.PrecheltStopping(verbose=False)

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

    loss_plot.set_loss(epoch,
                       train_loss / train_batches,
                       test_loss / test_batches)

    if stoppage.is_converged(test_loss / test_batches):
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
