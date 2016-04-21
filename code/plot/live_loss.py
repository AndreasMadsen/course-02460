
import os
import platform

import matplotlib.pyplot as plt
import numpy as np

class LiveLoss:
    def __init__(self, epochs, show=None):
        self.show = self.can_display() if show is None else show

        self._y_train = np.full(epochs, np.nan)
        self._y_test = np.full(epochs, np.nan)
        self._x_epochs = np.arange(1, epochs + 1)

        self._max_loss = 1

        if self.show:
            fig, self._ax = plt.subplots()
            self._train_points, = self._ax.plot(self._x_epochs, self._y_train, label='train')
            self._test_points, = self._ax.plot(self._x_epochs, self._y_test, label='test')
            plt.ylim(0, self._max_loss)
            plt.xlim(1, epochs)
            plt.legend()
            plt.ion()

    @staticmethod
    def can_display():
        if platform.system() == 'Windows':
            return True
        else:
            return 'DISPLAY' in os.environ and len(os.environ['DISPLAY']) > 0

    def set_loss(self, epoch, train, test):
        self._y_train[epoch] = train
        self._y_test[epoch] = test
        self._max_loss = max(self._max_loss, train, test)

        if self.show:
            # Update data
            self._train_points.set_data(self._x_epochs, self._y_train)
            self._test_points.set_data(self._x_epochs, self._y_test)
            # Update ylim
            self._ax.set_ylim([0, self._max_loss])
            # Allow time for interaction and update
            plt.pause(0.1)

    def finish(self):
        if self.show:
            plt.ioff()
            plt.show()
