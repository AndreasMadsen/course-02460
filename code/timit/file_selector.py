
import re
import os
import os.path as path
import textwrap

thisdir = path.dirname(path.realpath(__file__))
timitdir = path.realpath(path.join(thisdir, "..", "..", "data"))

usage_all = ["test", "train"]
dialect_all = ["dr1", "dr2", "dr3", "dr4", "dr5", "dr6", "dr7", "dr8"]
sex_all = ["f", "m"]
texttype_all = ["SA", "SI", "SX"]

def _tolist(value):
    return value if isinstance(value, (list, tuple)) else [value]

class FileSelector:
    def __init__(self, usage=None, dialect=None, sex=None, texttype=None):
        """Creates an iterable for iterating over all data files.

        It is possibol to select a subset of the files by settings an optional
        parameter. They can attain the values:
        usage    : "train" or "test"
        dialect  : "drX" where X \in [1, 8]
        sex      : "f" (female) or "m" (male)
        texttype : "SA", "SI" or "SX"
        """
        # Default parameters to use all combinations
        self.usage    = usage_all if (usage is None) else _tolist(usage)
        self.dialect  = dialect_all if (dialect is None) else _tolist(dialect)
        self.sex      = sex_all if (sex is None) else _tolist(sex)
        self.texttype = texttype_all if (texttype is None) else _tolist(texttype)

        self._files = list(self._file_list_generator())

    def _file_list_generator(self):
        for usage in self.usage:
            for dialect in self.dialect:
                # get all speakers based on usage and dialect
                dialect_dir = path.join(timitdir, usage, dialect)
                speakers = os.listdir(dialect_dir)

                for speaker in speakers:
                    for sex in self.sex:
                        # check that speaker has the correct sex
                        if (speaker[0] != sex): continue

                        # Get all sentences said by the speaker
                        speaker_dir = path.join(dialect_dir, speaker)
                        sentences = os.listdir(speaker_dir)

                        # Remove the file extension from the sentences filename
                        # and remove dublicates to get a unique list list of
                        # sentences identifiers.
                        sentences = set(map(
                            lambda name: re.sub('_?\.[A-Z]{3}', "", name),
                            sentences
                        ))

                        for sentence in sentences:
                            for texttype in self.texttype:
                                # check that sentence type is correct
                                if (sentence[0:2] != texttype): continue

                                sentence_file = path.join(speaker_dir, sentence)

                                yield DataItem(usage, dialect, sex, speaker,
                                               sentence, texttype,
                                               sentence_file)

    def __len__(self):
        return len(self._files)

    def __iter__(self):
        # TODO(Andreas): consider implementing this as an actual iterator
        # and not just returning a list.
        return iter(self._files)

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

if __name__ == '__main__':
    for file in FileSelector(usage='train', sex='f', dialect='dr1',
                             texttype='SI'):
        print(file)
