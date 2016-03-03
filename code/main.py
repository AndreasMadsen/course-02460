
from __future__ import print_function

import sys
import os
import numpy as np
import time

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from load_data import DataModel

def main():

    # Initialize data model
    data_model = DataModel()

    #data_dict = data_model.data_dict

    #print(data_model.classes)
    #print(data_model.class2vec(class_value='m'))
    #print(data_model.class2vec(class_value='f'))

    data_model.load_data()

    #Z = data_dict['test']['dr1']['mrjo0']['spectogram']

    # Load data


if __name__ == '__main__':
    main()
