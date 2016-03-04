
import matplotlib.pyplot as plt
import scipy.io.wavfile

(rate, data) = scipy.io.wavfile.read('./data/train/dr1/fcjf0/SA1_.WAV')

plt.specgram(data)
plt.show()
