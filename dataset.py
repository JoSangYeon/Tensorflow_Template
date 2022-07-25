import os
import math
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MyDataLoader(keras.utils.Sequence):
    def __init__(self, x_list, y_list, batch_size, shuffle=False,):
        super(MyDataLoader, self).__init__()
        self.x = x_list
        self.y = y_list

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]

        batch_x = np.array([self.x[i] for i in indices])
        batch_y = np.array([self.y[i] for i in indices])

        return batch_x, batch_y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indices)