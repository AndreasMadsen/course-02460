
import numpy as np

import timit
import helpers

# Create data selector object
selector = timit.FileSelector()
selector = helpers.TargetType(selector, target_type='speaker')
speakers = selector.labels
selector = helpers.Filter(selector, min_count=10, min_size=300, nperseg=256, noverlap=128)
selector = helpers.Spectrogram(selector, nperseg=256, noverlap=128, normalize_signal=True)
selector = helpers.Truncate(selector, truncate=300, axis=2)
selector = helpers.Normalize(selector)
selector = helpers.Validation(selector, test_fraction=0.25, stratified=True)

train_selector = helpers.Minibatch(selector.train)
test_selector  = helpers.Minibatch(selector.test)

# Count speakers
speakers_count = np.zeros(shape=len(speakers))
speakers_count_train = np.zeros(shape=len(speakers))
speakers_count_test  = np.zeros(shape=len(speakers))

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
    if (count == 0):
        print('Speaker train: %s was removed in the filter function' % (speaker))
    else:
        print('Speaker train: %s, count: %d' % (speaker, count))
print('')
for speaker, count in zip(speakers, speakers_count_train):
    if (count == 0):
        print('Speaker test: %s was removed in the filter function' % (speaker))
    else:
        print('Speaker test: %s, count: %d' % (speaker, count))
print('')

# Remove speakers with 0 counts.
speakers_count = np.delete(speakers_count, np.where(speakers_count == 0))
speakers_count_train = np.delete(speakers_count_train, np.where(speakers_count_train == 0))
speakers_count_test = np.delete(speakers_count_test, np.where(speakers_count_test == 0))

# Other details
print('Number of speakers: %d' % (len(speakers)))
print('Number of active speakers: %d' % (len(speakers_count)))
print('Number of observations filtered: %d' % (np.sum(speakers_count)))

print('min speaker count: %d' % (np.min(speakers_count)))
print('min speaker count train: %d' % (np.min(speakers_count_train)))
print('min speaker count test: %d' % (np.min(speakers_count_test)))

print('max speaker count: %d' % (np.max(speakers_count)))
print('max speaker count train: %d' % (np.max(speakers_count_train)))
print('max speaker count test: %d' % (np.max(speakers_count_test)))

print('mean speaker counts: %d' % (np.mean(speakers_count)))
