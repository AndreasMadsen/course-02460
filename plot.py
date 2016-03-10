
from __future__ import print_function

import os
filepath = os.path.dirname(os.path.abspath(__file__))

import matplotlib.pyplot as plt
import numpy as np



def main():

    # Load training errors
    train_errs = np.genfromtxt(os.path.join(filepath, 'errors/%s.csv' % ('train_err')), delimiter=',')
    epochs = np.arange(0, train_errs.shape[0])


    plt.plot(epochs, train_errs, label='Training error')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Error')

    # Save figure
    plt.savefig(os.path.join(filepath, 'errors/%s.eps' % ('train_err')), format='eps')
    plt.savefig(os.path.join(filepath, 'errors/%s.png' % ('train_err')), format='png')

    plt.show()




if __name__ == '__main__':
    main()
