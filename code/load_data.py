
from __future__ import print_function

import sys
import os
import csv
import pickle
import numpy as np
filepath = os.path.dirname(os.path.abspath(__file__))

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.io import wavfile

from sklearn.cross_validation import train_test_split

class DataModel():
    # Static variables
    DATA_SET_PATH   = os.path.join(filepath, 'data/timit/')
    DATA_SET_FILE_DATA_DICT = os.path.join(filepath, 'data/data_dict')
    DATA_SET_FILE_X = os.path.join(filepath, 'data/X')
    DATA_SET_FILE_Y = os.path.join(filepath, 'data/Y')

    _data_dict = None
    _class_type = None
    _classes = None
    _X = None
    _Y = None

    def __init__(self, reset=False, class_type='gender'):
        self._class_type=class_type

        if reset:
            print("Reset forced!")
            for f in [
                self.DATA_SET_FILE_DATA_DICT,
                self.DATA_SET_FILE_X,
                self.DATA_SET_FILE_Y
            ]:
                try:
                    os.remove(f)
                except:
                    pass

    @property
    def data_dict(self):
        if self._data_dict is not None:
            return self._data_dict

        try:
            print("Loading data dictionary..")
            self.load_data_dict()
        except:
            print("Pickled file not found..")
            print("Creating data dictionary from files..")
            self._data_dict = self.create_data_dict()
            print("Pickling file..")
            self.save_data_dict()
        return self._data_dict

    def create_data_dict(self):
        """
            Returns dictionary for a given usage
        """
        data_dict = dict()
        for usage in ['train', 'test']:
            data_dict[usage] = dict()
            dialect_folders = [os.path.join(self.DATA_SET_PATH,'%s/%s' % (usage, f)) \
                for f in os.listdir(os.path.join(self.DATA_SET_PATH, '%s/' % (usage)))]
            dialect_folders = filter(lambda x: not os.path.isfile(x), dialect_folders)
            for dialect_folder in dialect_folders:
                dialect = os.path.basename(dialect_folder)
                data_dict[usage][dialect] = dict()
                speaker_folders = [os.path.join(dialect_folder, f) for f in os.listdir(dialect_folder)]
                for speaker_folder in speaker_folders:
                    speaker = os.path.basename(speaker_folder)
                    data_dict[usage][dialect][speaker] = dict()
                    files = [os.path.join(speaker_folder, f) for f in os.listdir(speaker_folder)]
                    files = filter(lambda x: os.path.isfile(x), files)
                    for f in files:
                        tmp, f_ext = os.path.splitext(f)
                        f_name = '.'.join(os.path.basename(f).split('.')[0:-1])
                        # Don't use .wav files without '_' in the file (has no header information)
                        if '_' not in f_name:
                            continue

                        if not f_name in data_dict[usage][dialect][speaker].keys():
                            data_dict[usage][dialect][speaker][f_name] = dict()
                        if   f_ext.lower() == '.wav':
                            # Read .wav file
                            with open(f, 'rb') as _f:
                                sample_rate, wav_dat = wavfile.read(_f, mmap=False) # Maybe use mmap True
                                data_dict[usage][dialect][speaker][f_name]['sample_rate'] = sample_rate
                                data_dict[usage][dialect][speaker][f_name]['data_raw'] = wav_dat
                                # Compute spectogram
                                data_dict[usage][dialect][speaker][f_name]['spectogram'] = \
                                    self.compute_spectogram(x=wav_dat, Fs=sample_rate)

                        elif f_ext.lower() == '.phn':
                            # Time-aligned phonetic transcription
                            data_dict[usage][dialect][speaker][f_name]['phn'] = []
                            with open(f, 'rb') as _f:
                                reader = csv.reader(_f, delimiter=' ')
                                for row in reader:
                                    data_dict[usage][dialect][speaker][f_name]['phn'].append(row)
                                data_dict[usage][dialect][speaker][f_name]['phn'] = \
                                    np.array(data_dict[usage][dialect][speaker][f_name]['phn'])

                        elif f_ext.lower() == '.txt':
                            # Associated orthographic transcription
                            with open(f, 'rb') as _f:
                                for row in _f:
                                    data_dict[usage][dialect][speaker][f_name]['ort'] = \
                                        np.array(row.split(' '))

                        elif f_ext.lower() == '.wrd':
                            # Time-aligned word transcription
                            data_dict[usage][dialect][speaker][f_name]['wrd'] = []
                            with open(f, 'rb') as _f:
                                reader = csv.reader(_f, delimiter=' ')
                                for row in reader:
                                    data_dict[usage][dialect][speaker][f_name]['wrd'].append(row)
                                data_dict[usage][dialect][speaker][f_name]['wrd'] = \
                                    np.array(data_dict[usage][dialect][speaker][f_name]['wrd'])
        return data_dict

    def save_data_dict(self):
        with open(self.DATA_SET_FILE_DATA_DICT, 'wb') as f:
            pickle.dump(self.data_dict, f)

    def load_data_dict(self):
        with open(self.DATA_SET_FILE_DATA_DICT, 'rb') as f:
            self._data_dict = pickle.load(f)

    def compute_spectogram(self, x, Fs, NFFT=256, noverlap=128):
        """
            Computes the spectogram data image, which is being plotted by the
            pyplot.specgram function, from a given audio track `x` and a sample
            rate `Fs`.
        """
        spec, freqs, t, im = plt.specgram(x, Fs=Fs, NFFT=NFFT, noverlap=noverlap)
        spec = 10. * np.log10(spec)
        return np.flipud(spec).astype('float32')

    @property
    def classes(self):
        if self._classes is None:
            self.create_classes()
        return self._classes

    def create_classes(self):
        """
            Returns the unique classes given in the dataset
        """
        if self._class_type == 'gender':
            first_letters = []
            for usage in self.data_dict.keys():
                for dialect in self.data_dict[usage]:
                    for speaker in self.data_dict[usage][dialect]:
                        first_letters.append(speaker[0])
            self._classes = np.unique(np.array(first_letters))

    def class2vec(self, class_value):
        return (self.classes == class_value).astype('int')

    def texts_generator(self):
        for usage in self.data_dict.keys():
            for dialect in self.data_dict[usage].keys():
                for speaker in self.data_dict[usage][dialect].keys():
                    for text_type in self.data_dict[usage][dialect][speaker].keys():
                        yield (usage, dialect, speaker, text_type, \
                               self.data_dict[usage][dialect][speaker][text_type])

    @property
    def data(self):
        if self._X is not None and self._Y is not None:
            return self._X['train'], self._Y['train'], \
                   self._X['val'],   self._Y['val'], \
                   self._X['test'],  self._Y['test']
        try:
            print("Loading data files..")
            self.load_data_files()
        except:
            print("Pickled file not found..")
            print("Creating data files from dictionary..")
            self._X, self._Y = self.create_data()
            print("Pickling files..")
            self.save_data_files()
        return self._X['train'], self._Y['train'], \
               self._X['val'],   self._Y['val'], \
               self._X['test'],  self._Y['test']


    def create_data(self, time_windows=384):
        """
            `time_windows` = length of window
        """
        X, Y = {}, {}
        for usage, dialect, speaker, text_type, data in self.texts_generator():

            # Very naive way of taking only 3 second windows of spectogram
            if data['spectogram'].shape[1] < time_windows:
                continue

            x = data['spectogram'][:,0:time_windows]
            x = x.reshape((1, 1, x.shape[0], x.shape[1]))
            y = self.class2vec(class_value=speaker[0])

            if usage not in X.keys():
                X[usage] = x
            else:
                X[usage] = np.vstack((X[usage], x))

            if usage not in Y.keys():
                Y[usage] = []
            Y[usage].append(y)

        for usage in Y.keys():
            Y[usage] = np.array(Y[usage]).astype('int32')

        X['train'], X['val'], Y['train'], Y['val'] = train_test_split(X['train'], Y['train'], test_size=0.10, random_state=42)
        return X, Y

    def save_data_files(self):
        with open(self.DATA_SET_FILE_X, 'wb') as f:
            pickle.dump(self._X, f)
        with open(self.DATA_SET_FILE_Y, 'wb') as f:
            pickle.dump(self._Y, f)

    def load_data_files(self):
        with open(self.DATA_SET_FILE_X, 'rb') as f:
            self._X = pickle.load(f)
        with open(self.DATA_SET_FILE_Y, 'rb') as f:
            self._Y = pickle.load(f)

def main():
    data_model = DataModel(reset=True)
    data_dict = data_model.data_dict
    del data_dict
    data = data_model.data
    del data
    #print(data_dict)
    #print(data_dict.keys())
    #print(data_dict['test'].keys())
    #print(data_dict['test']['dr1'].keys())
    #print(data_dict['test']['dr1']['mrjo0'].keys())



    #spectogram = data_model.compute_spectogram(x=x, Fs=Fs)

if __name__ == '__main__':
    main()
