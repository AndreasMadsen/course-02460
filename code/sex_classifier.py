
import os
import numpy as np
import lasagne
import theano
import theano.tensor as T

import timit
import network
import helpers
import plot

def create_selector(usage):
    selector = timit.FileSelector(usage=usage)#,dialect='dr1')
    selector = helpers.TargetType(selector, target_type='sex')
    selector = helpers.Spectrogram(selector, nperseg=256, noverlap=128, normalize_signal=True)
    selector = helpers.Truncate(selector, truncate=300, axis=2)
    selector = helpers.Normalize(selector)
    selector = helpers.Minibatch(selector)
    return selector

test_selector = create_selector('test')
train_selector = create_selector('train')

cnn = network.DielemanCNN(input_shape=(1, 129, 300), output_units=2,
		          regularization=1e-1, verbose=True)
cnn.compile()

epochs = 500

loss_array =  np.zeros((epochs, 2))

# implement early stopping as in http://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf
# using the second definition of a stopping criteria from the paper
# NOTE: we are not using classical early stopping since we run the criteria on the training set

early_stop = 0.5
strip_length = 10

loss_plot = plot.LiveLoss(epochs)

print("Training with early stopping. ",end="")
print("Criterion is [generalization loss] / [improvement factor] > alpha ")
print("\talpha = {0}".format(early_stop))
print("\timprovement factor length = {0}".format(strip_length))
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
    
    loss_array[epoch, 0] = train_loss / train_batches
    loss_array[epoch, 1] = test_loss  / test_batches
       
    print("Epoch %d: Train Loss %g, Test Loss %g" % (
          epoch + 1, loss_array[epoch, 0],loss_array[epoch, 1]))

    loss_plot.set_loss(epoch,
                       loss_array[epoch, 0],
                       loss_array[epoch, 1])
    if epoch == 0:
        lowest_loss_val = 1e10
    else:
        lowest_loss_val = min(loss_array[:epoch, 0])
    
    generalization_loss = loss_array[epoch, 0] / lowest_loss_val - 1
    
    if epoch < strip_length:
        improvement_factor = 1
    else:
        strip = loss_array[(epoch - strip_length):epoch, 0]
        improvement_factor = np.mean(strip) / np.min(strip) - 1
    
    criterion = generalization_loss / improvement_factor 
    #print("gen_loss = {0:.3f}".format(generalization_loss))
    #print("improvement_factor = {0:.3f}".format(improvement_factor))
    if criterion > early_stop:
        print("Stopping early because gl / if  = {0:.3f}>{1:.3f} ".format(criterion, early_stop))
        break

missclassifications = 0
observations = 0
for (test_input, test_target) in test_selector:
    predict = np.argmax(cnn.predict(test_input), axis=1)
    observations += len(predict)
    missclassifications += np.sum(predict != test_target)

print('missrate: %f' % (missclassifications / observations))

loss_plot.finish()
