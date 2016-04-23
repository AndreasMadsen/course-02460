
import os
import os.path as path
import random

from elsdsr.data_item import DataItem

thisdir = path.dirname(path.realpath(__file__))
elsdsrdir = path.realpath(path.join(thisdir, "..", "..", "data", "elsdsr"))

usage_all = ["test", "train"]
sex_all = ["F", "M"]
speaker_all = ["ASM", "CBR", "FKC", "KBP", "LKH", "MLP", "MNA", "NHP", "OEW", \
               "PRA", "REM", "RKO", "TLS", "AML", "DHH", "EAB", "HRO", "JAZ", \
               "MEL", "MEV", "SLJ", "TEJ", "UAN"]

# TODO: Implement filtering for specific paragraph and sentence
# S = train
# Sr = Test
texttype_all = ["S", "Sr"]
paragraphs = ['a', 'b', 'c', 'd', 'e', 'f', 'g'] # Train
sentence_number = list(range(1, 47)) # Test

def _tolist(value):
    return value if isinstance(value, (list, tuple)) else [value]

class FileSelector:
    def __init__(self, usage=None, sex=None, speaker=None, shuffle=True):
        """Creates an iterable for iterating over all data files.

        It is possibol to select a subset of the files by settings an optional
        parameter. They can attain the values:
        usage    : "train" or "test"
        sex      : "f" (female) or "m" (male)
        speaker  : "ASM", "CBR", "FKC", "KBP", "LKH", "MLP", "MNA", "NHP", "OEW",
                   "PRA", "REM", "RKO", "TLS", "AML", "DHH", "EAB", "HRO", "JAZ",
                   "MEL", "MEV", "SLJ", "TEJ" or "UAN"
        """
        # Default parameters to use all combinations
        self.usage    = usage_all if (usage is None) else _tolist(usage)
        self.sex      = sex_all if (sex is None) else _tolist(sex)
        self.sex      = list(map(lambda x: x.upper(), self.sex))
        self.speakers = speaker_all if (speaker is None) else _tolist(speaker)

        self._files = list(self._file_list_generator())

        if (shuffle):
            random.shuffle(self._files)

    def _file_list_generator(self):
        for usage in self.usage:
            # get all speakers based on usage and dialect
            usage_dir = path.join(elsdsrdir, usage)
            speaker_files = os.listdir(usage_dir)

            for speaker_file in speaker_files:
                # Extract values
                sex       = speaker_file[0]
                speaker   = speaker_file[1:4]
                file_path = path.join(usage_dir, speaker_file)

                # Chech that speaker has the correct sex
                if sex not in self.sex: continue

                # Check that speaker is allowed
                if speaker not in self.speakers: continue

                yield DataItem(usage, sex, speaker, file_path)

    def __len__(self):
        return len(self._files)

    def __iter__(self):
        # TODO(Andreas): consider implementing this as an actual iterator
        # and not just returning a list.
        return iter(self._files)


if __name__ == '__main__':
    for file in FileSelector(usage='train', sex='f', speaker=None):
        print(file)
