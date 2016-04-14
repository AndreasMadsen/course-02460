
import numpy as np

import timit
import helpers

# Create data selector object
selector = timit.FileSelector(dialect=None)
selector = helpers.TargetType(selector, target_type='speaker')
speakers = selector.labels
selector = helpers.Spectrogram(selector, nperseg=256, noverlap=128, normalize_signal=True)
selector = helpers.Truncate(selector, truncate=300, axis=2)
selector = helpers.Normalize(selector)
selector = helpers.Validation(selector, test_fraction=0.25, stratified=True)

train_selector = helpers.Minibatch(selector.train)
test_selector  = helpers.Minibatch(selector.test)

# TODO: Count in train and test

# Count speakers
speakers_count = np.zeros(shape=len(speakers.keys()))
speakers_count_train = np.zeros(shape=len(speakers.keys()))
speakers_count_test  = np.zeros(shape=len(speakers.keys()))

for _, target in train_selector:
    for item in target:
        speakers_count_train[item] += 1
        speakers_count[item] += 1

for _, target in test_selector:
    for item in target:
        speakers_count_test[item] += 1
        speakers_count[item] += 1

# Print speakers count result
for speaker, count in zip(speakers, speakers_count_test):
    print('Speaker train: %s, count: %d' % (speaker, count))
print('')
for speaker, count in zip(speakers, speakers_count_train):
    print('Speaker test: %s, count: %d' % (speaker, count))
print('')

# Other details
print('Number of speakers: %d' % (len(speakers)))
print('min speaker count: %d' % (np.min(speakers_count)))
print('min speaker count train: %d' % (np.min(speakers_count_train)))
print('min speaker count test: %d' % (np.min(speakers_count_test)))
print('max speaker count: %d' % (np.max(speakers_count)))
print('max speaker count train: %d' % (np.max(speakers_count_train)))
print('max speaker count test: %d' % (np.max(speakers_count_test)))
print('mean speaker counts: %d' % (np.mean(speakers_count)))
