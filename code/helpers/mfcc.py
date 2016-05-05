
from features import mfcc

class MFCC:
    def __init__(self, selector, normalize_signal=False, **kwargs):
        self._selector = selector
        self._normalize_signal = normalize_signal

    def __iter__(self):
        for item, target in self._selector:
            mfcc_feat = mfcc(signal=item.signal(normalize_signal=self._normalize_signal), samplerate=item.rate)
            yield (mfcc_feat[:, 1:], target)
