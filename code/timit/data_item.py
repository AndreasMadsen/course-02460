
import scipy
import os.path as path
import textwrap

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

    def signal(self):
        (_, signal) = scipy.io.wavfile.read(self.wav)
        return signal

    def spectogram(self, nfft=256, noverlap=128):
        signal = self.signal()
        (_, _, spectrogram) = scipy.signal.spectrogram(signal, 1 / self.rate,
                                                       nfft=nfft,
                                                       noverlap=noverlap)
        return spectrogram
