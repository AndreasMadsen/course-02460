
import numpy as np

class MinibatchSpectogram:
    def __init__(self, selector, batchsize=5, fixed_width=250):
        self.selector = selector
        self.batchsize = batchsize
        self.fixed_width = fixed_width

    def __iter__(self):
        return MinibatchSpectogramIterator(self.selector, self.batchsize, self.fixed_width)

class MinibatchSpectogramIterator:
    def __init__(self, selector, batchsize, fixed_width):
        self.selector = iter(selector)
        self.batchsize = batchsize
        self.fixed_width = fixed_width

        self._no_more_data = False

    def __next__(self):
        # If last batch drained the selector, then stop iteration now
        if (self._no_more_data): raise StopIteration

        # Collect `batchsize` items or until the selector is drained
        num_items = 0
        input_items = []
        target_items = []
        while (num_items < self.batchsize):
            try:
                item = next(self.selector)
            except StopIteration:
                self._no_more_data = True
                break

            spectogram = item.spectogram()

            if (spectogram.shape[1] >= self.fixed_width):
                spectogram = spectogram[:, 0:self.fixed_width]
                spectogram = spectogram.reshape(1, *spectogram.shape)

                input_items.append(spectogram)
                target_items.append(int(item.sex == 'f'))
                num_items += 1

        # If no data was collected stop iteration
        if (num_items == 0): raise StopIteration

        # If some data was collected return that as a batch
        return (np.asarray(input_items), np.asarray(target_items, dtype='int32'))