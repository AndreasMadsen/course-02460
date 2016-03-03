
from __future__ import print_function

import sys
import os
import csv
import pickle
import numpy as np
filepath = os.path.dirname(os.path.abspath(__file__))

from scipy.io import wavfile

from sklearn.feature_extraction.text import CountVectorizer

class DataModel():
    # Static variables
    DATA_SET_PATH   = os.path.join(filepath, 'data/timit/')
    DATA_SET_FILE_X = os.path.join(filepath, 'data/X.npy')
    DATA_SET_FILE_Y = os.path.join(filepath, 'data/Y.npy')

    def __init__(self):
        self._data_dict = None

    @property
    def data_dict(self):
        if self._data_dict is None:
            self._data_dict = self.create_data_dict()

        return self._data_dict

    def create_data_dict(self):
        """
            Returns dictionary for a given usage
        """
        data_dict = dict()
        for usage in ['train', 'test']:
            data_dict[usage] = dict()
            dialect_folders = [os.path.join(self.DATA_SET_PATH,'%s/%s' % (usage, f)) for f in os.listdir(os.path.join(self.DATA_SET_PATH, '%s/' % (usage)))]
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
                        f_name = os.path.basename(f)
                        if   f_ext.lower() == '.wav':
                            # Don't use .wav files without '_' in the file (has no header information)
                            if '_' not in f_name:
                                continue
                            # Read .wav file
                            with open(f, 'rb') as _f:
                                sample_rate, wav_dat = wavfile.read(_f, mmap=False) # Maybe use mmap True
                                data_dict[usage][dialect][speaker]['sample_rate'] = sample_rate
                                data_dict[usage][dialect][speaker]['data'] = wav_dat
                        elif f_ext.lower() == '.phn':
                            # Time-aligned phonetic transcription
                            data_dict[usage][dialect][speaker]['phn'] = []
                            with open(f, 'rb') as _f:
                                reader = csv.reader(_f, delimiter=' ')
                                for row in reader:
                                    data_dict[usage][dialect][speaker]['phn'].append(row)
                                data_dict[usage][dialect][speaker]['phn'] = np.array(data_dict[usage][dialect][speaker]['phn'])
                        elif f_ext.lower() == '.txt':
                            # Associated orthographic transcription
                            with open(f, 'rb') as _f:
                                for row in _f:
                                    data_dict[usage][dialect][speaker]['ort'] = np.array(row.split(' '))
                        elif f_ext.lower() == '.wrd':
                            # Time-aligned word transcription
                            data_dict[usage][dialect][speaker]['wrd'] = []
                            with open(f, 'rb') as _f:
                                reader = csv.reader(_f, delimiter=' ')
                                for row in reader:
                                    data_dict[usage][dialect][speaker]['wrd'].append(row)
                                data_dict[usage][dialect][speaker]['wrd'] = np.array(data_dict[usage][dialect][speaker]['wrd'])
        return data_dict

    def save_data_dict(self):
        data_dict = self.get_data_dict()


# Persistency
def save_file(X, filename):
    with open(filename, 'wb') as f:
        pickle.dump(X, f)
def load_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def main():
    data_model = DataModel()
    data_dict = data_model.data_dict
    print(data_dict)
    print(data_dict.keys())
    print(data_dict['test'].keys())
    print(data_dict['test']['dr1'].keys())
    print(data_dict['test']['dr1']['mrjo0'].keys())
    print(data_dict['test']['dr1']['mrjo0']['data'])

    #save_file(X=data_dict, filename=)

if __name__ == '__main__':
    main()
