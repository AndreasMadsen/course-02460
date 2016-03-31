
import numpy as np
import lasagne
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

import timit
import network
import helpers

cnn = network.SimpleCNN(input_shape=(1, 129, 300), output_units=2, verbose=True)
cnn.compile(all_layers=True)

def create_selector(usage):
    selector = timit.FileSelector(usage=usage, dialect='dr1')
    selector = helpers.Spectrogram(selector, nperseg=256, noverlap=128, normalize_signal=True)
    selector = helpers.Truncate(selector, truncate=300, axis=2)
    selector = helpers.Normalize(selector)
    selector = helpers.Minibatch(selector, cache=True)
    return selector

test_selector = create_selector('test')

for input, target in test_selector:
    outputs = cnn.all_layer_outputs(input)

    plt.figure()
    for i, layer_output in enumerate(outputs):
        plt.subplot(len(outputs), 1, i + 1)
        layer_output = layer_output.ravel()
        print(np.min(layer_output), np.max(layer_output))
        plt.hist(layer_output,
                 bins=np.linspace(np.min(layer_output), np.max(layer_output), 100),
                 normed=True)

    break

plt.show()
