
import os.path as path
import textwrap

import numpy as np
import scipy
import scipy.io.wavfile
import scipy.signal

thisdir = path.dirname(path.realpath(__file__))
timitdir = path.realpath(path.join(thisdir, "..", "..", "data", "timit"))

class DataItem:
    def __init__(self, usage, dialect, sex, speaker, sentence, texttype,
                 filename):
        self.file = filename[len(timitdir):]
        self.usage = usage
        self.dialect = dialect
        self.sex = sex
        self.speaker = speaker
        self.sentence = sentence
        self.texttype = texttype

        self.wav = filename + "_.WAV"
        self.phn = filename + ".PHN"
        self.txt = filename + ".TXT"
        self.wrd = filename + ".WRD"

        self._signal_shape = None
        self.rate = 16000  # This is the same for all files in timit

    def __str__(self):
        return textwrap.dedent("""\
        file: {file}
            - usage: {usage}
            - dialect: {dialect}
            - sex: {sex}
            - speaker: {speaker}
            - sentence: {sentence}
            - texttype: {texttype}
        """.format(**vars(self)))

    @property
    def signal_shape(self):
        if self._signal_shape is None:
            (_, signal) = scipy.io.wavfile.read(self.wav, mmap=True)
            self._signal_shape = signal.shape
        return self._signal_shape

    def spectrogram_shape(self, nperseg=256, noverlap=128):
        # https://github.com/scipy/scipy/blob/master/scipy/signal/spectral.py#L846
        if nperseg % 2:
            num_freqs = (nperseg + 1) // 2
        else:
            num_freqs = nperseg // 2 + 1

        # https://github.com/scipy/scipy/blob/master/scipy/signal/spectral.py#L878
        t = np.arange(nperseg / 2,
                      self.signal_shape[-1] - nperseg / 2 + 1,
                      nperseg - noverlap)
        num_times = t.shape[0]

        return (num_freqs, num_times)

    def size(self, *args, **kwargs):
        return self.spectrogram_shape(*args, **kwargs)[1]

    def signal(self, normalize_signal=False):
        (_, signal) = scipy.io.wavfile.read(self.wav)
        self._signal_shape = signal.shape

        # There is an issue in numpy where np.linalg.norm can overflow for
        # 16bit ints. Thus the signal is converted to float32 if normalized
        # https://github.com/numpy/numpy/issues/6128
        if (normalize_signal):
            signal = signal.astype('float32')
            signal = signal / np.linalg.norm(signal)

        return signal

    def spectrogram(self, normalize_signal=False, normalize_spectogram=False,
                    *args, **kwargs):
        signal = self.signal(normalize_signal=normalize_signal)
        (_, _, spectrogram) = scipy.signal.spectrogram(signal, fs=self.rate,
                                                       *args, **kwargs)

        if (normalize_spectogram):
            spectrogram = spectrogram / np.linalg.norm(spectrogram, 'fro')

        return spectrogram
