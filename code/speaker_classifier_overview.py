
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
selector = helpers.Validation(selector, test_fraction=0.25)
train_selector = helpers.Minibatch(selector.train)
test_selector  = helpers.Minibatch(selector.test)

# Count speakers
speakers_count = np.zeros(shape=len(speakers.keys()))
for selector in [train_selector, test_selector]:
    for _, target in selector:
        for item in target:
            speakers_count[item] += 1

# Print speakers count result
for speaker, count in zip(speakers, speakers_count):
    print('Speaker: %s, count: %d' % (speaker, count))
print('')

# Other details
print('Number of speakers: %d' % (len(speakers)))
print('min speaker count: %d' % (np.min(speakers_count)))
print('max speaker count: %d' % (np.max(speakers_count)))
print('mean speaker counts: %d' % (np.mean(speakers_count)))
