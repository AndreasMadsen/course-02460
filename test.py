
from load_data import DataModel

from scipy.io import wavfile
import numpy as np


data_model = DataModel()



with open('data/timit/train/dr1/fcjf0/SA1_.WAV') as _f:
    sample_rate, x = wavfile.read(_f, mmap=False)

#spec = data_model.compute_spectogram(x=x, Fs=sample_rate, noverlap=128)
spec = data_model.compute_spectogram(x=x, Fs=256, noverlap=128)
print(spec)

print(np.max(spec))
