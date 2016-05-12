
import os
import numpy as np
import lasagne
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

import timit
import elsdsr
import network
import helpers
import early_stopping

# Create data selector object
nfolds = 20
selector = timit.FileSelector()
selector = helpers.TargetType(selector, target='sex')
selector = helpers.Spectrogram(selector, nperseg=256, noverlap=128, normalize_signal=True)
selector = helpers.Splitter(selector, split_size=100, axis=2)
selector = helpers.Normalize(selector)
selector = helpers.CrossValidation(selector, folds=nfolds, stratified=True)

#reg_values = [1] + [x* 10 ** (-y) for y in range(1, 11) for x in [5, 2, 1]] + [0]
reg_values = [1] + [x* 10 ** (-y) for y in range(1, 11) for x in [1]] + [0]
#reg_values = [0.1, 0.5]

missrates = np.zeros((nfolds, len(reg_values)))
for k, fold in enumerate(selector.folds):
    train_selector = helpers.Minibatch(fold.train)
    test_selector  = helpers.Minibatch(fold.test)

    for i, reg_val in enumerate(reg_values):
        cnn = network.DielemanCNN(input_shape=(1, 129, 100), output_units=len(speakers),
                                  verbose=True, learning_rate=0.001, dropout=False)
        #cnn = network.Logistic(input_shape=(1, 129, 100), output_units=len(speakers), verbose=True)
        cnn.add_regularizer(network.regularizer.WeightDecay(reg_val))
        cnn.compile()

        epochs = 500
        #epochs = 10

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

            if stoppage.is_converged(test_loss / test_batches):
                print("Stopping early")
                break

        missclassifications = 0
        observations = 0
        for (test_input, test_target) in test_selector:
            predict = np.argmax(cnn.predict(test_input), axis=1)
            observations += len(predict)
            missclassifications += np.sum(predict != test_target)

        missrate = missclassifications / observations
        missrates[k,i] = missrate

reg_values = np.array(reg_values)

X = np.vstack((reg_values, missrates))

# Save values
np.savetxt('./output/reg_opt_dieleman_weight_decay_sex.csv', X)
