
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

    def signal(self, normalize_signal=False):
        (_, signal) = scipy.io.wavfile.read(self.wav)

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
