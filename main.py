### Ronik and Dawson's attempt at machine learning
### yay!!!

import numpy as np
import tensorflow as tf
from tensorflow import keras
import csv

from tensorflow.keras.layers.experimental.preprocessing import Normalization


class WineRater:
    def __init__(self):
        redData, whiteData = self.getWineQualityData()
        self.redWineData = redData
        self.whiteWineData = whiteData
        self.normalizeData()

    def getWineQualityData(self):
        """
        Returns two numpy arrays that contain the data.
        """
        with open("winequality-red.csv") as file:
            # Ignore the first line since it's text
            file.readline()
            red = np.loadtxt(file, delimiter=";")

        with open("winequality-white.csv") as file:
            # Ignore the first line since it's text
            file.readline()
            white =  np.loadtxt(file, delimiter=";")

        return (red, white)

    def normalizeData(self):
        """
        Normalize each column of the data so that it has a standard
        variation of 1 and a mean of 0, as keras expects.
        """
        for array in (self.redWineData, self.whiteWineData):
            for i, column in enumerate(array.T):
                # generate normalization function
                normalizer = Normalization(axis=-1)
                normalizer.adapt(column)

                # mutate data
                normalized = normalizer(column)
                array[:, i] = np.reshape(normalized, column.shape)


if __name__ == "__main__":
    wr = WineRater()
    
