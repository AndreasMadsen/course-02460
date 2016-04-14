
import numpy as np

class Validation:
    def __init__(self, selector, test_fraction=0.33, stratified=False, **kwargs):
        self.selector = selector

        # Load all targets (labels)
        labels = [target for (_, target) in self.selector]

        # Create shuffled index array
        n = len(labels)
        self.n_split = int(n * test_fraction)

        # Create idx splitting mapping
        # 0 = Train
        # 1 = Test
        split_idx = {}

        if stratified:
            # Make label counter for train and test
            labels_unique = sorted(list(set(labels)))
            label_count_train = {label: 0 for label in labels_unique}
            label_count_test  = {label: 0 for label in labels_unique}

            for i in range(0, n):
                label = labels[i]

                # Make sure atleast 1 of each label is present in train and test
                if label_count_train[label] == 0:
                    split_idx[i] = 0
                    label_count_train[label] += 1
                    continue

                if label_count_test[label] == 0:
                    split_idx[i] = 1
                    label_count_test[label] += 1
                    continue

                # If one label already is present in train and test, add label
                # according to the fraction of labels in test.
                label_test_fraction = label_count_test[label] / (label_count_train[label] + label_count_test[label])

                if label_test_fraction > test_fraction:
                    split_idx[i] = 0
                    label_count_train[label] += 1
                else:
                    split_idx[i] = 1
                    label_count_test[label] += 1

        else:
            # Simple split
            for idx in self.indices:
                split_idx[idx] = int(idx < self.n_split)

        self.split_idx = split_idx

    @property
    def train(self):
        for i, (input, target) in enumerate(self.selector):
            if (self.split_idx[i] == 0):
                yield input, target

    @property
    def test(self):
        for i, (input, target) in enumerate(self.selector):
            if (self.split_idx[i] == 1):
                yield input, target
