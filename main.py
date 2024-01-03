"""
Barry Yu
Dec 28, 2023
Pytorch Speech Classification
"""
# ==================== IMPORTS ====================

# import packages
# IMPORTANT: make sure torch and torchaudio are 2.0.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt  # data visualization
import IPython.display as ipd
from tqdm import tqdm

# import data

from torchaudio.datasets import SPEECHCOMMANDS
import os


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


LABELS = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight',
    'five', 'follow', 'forward', 'four', 'go', 'happy', 'house',
    'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on',
    'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
    'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]

# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

# plt.plot(waveform.t().numpy())
# plt.show()
# labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
# print(labels)


# ==================== FORMATTING DATA ====================

# downsample the audio (lower resolution)
new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)


# encode each word as an index in LABELS
def label_to_index(_word):
    """
    # Return the position of the word in labels
    """
    return torch.tensor(LABELS.index(_word))


def index_to_label(_index):
    """
    Return the word corresponding to the index in labels
    This is the inverse of label_to_index
    """
    return LABELS[_index]


word_start = "no"
index = label_to_index(word_start)
word_recovered = index_to_label(index)

print(word_start, "-->", index, "-->", word_recovered)
