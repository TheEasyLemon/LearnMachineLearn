### Ronik and Dawson's attempt at machine learning
### yay!!!

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import csv

from tensorflow.keras.layers.experimental.preprocessing import Normalization


class WineRater:
    def __init__(self):
        self.inputLabels = ["fixed acidity", "volatile acidity", "citric acid",
                           "residual sugar", "chlorides", "free sulfur dioxide",
                           "total sulfur dioxide", "density", "pH", "sulphates",
                           "alcohol"]

        redData, whiteData = self.getWineQualityData()
        self.redWineData = redData
        self.whiteWineData = whiteData
        # self.normalizeData()

        redIO, whiteIO = self.splitData()
        # "Red Training Inputs" = "rTrainI"
        self.rTrainI, self.rTrainO, self.rTestI, self.rTestO = redIO
        self.wTrainI, self.wTrainO, self.wTestI, self.wTestO = whiteIO

        self.buildModel()
        self.trainModel()
        self.testModel()


    def getWineQualityData(self):
        """
        Returns two numpy arrays that contain the red/white wine data.
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

    def splitData(self):
        """
        Split the data into training/testing sets, and also
        split off the ratings (which is what we're trying
        to predict).
        Returns:
        [[redTrainingInputs, redTrainingOutputs, redTestingInputs,
        redTestingOutputs], [whiteTrainingInputs, whiteTrainingOutputs,
        whiteTestingInputs, whiteTestingOutputs]]

        Side effect is that the redWineData and whiteWineData is shuffled.
        """
        values = [None, None]

        for i, array in enumerate((self.redWineData, self.whiteWineData)):
            # Split into two (halfway down the file)
            np.random.shuffle(array)
            training, testing = np.split(array, [array.shape[0] // 2], axis=0)

            trainingInputs = training[:, :-1]
            trainingOutputs = training[:, -1]
            testingInputs = testing[:, :-1]
            testingOutputs = testing[:, -1]

            values[i] = [trainingInputs, trainingOutputs,
                         testingInputs, testingOutputs]

        return values

    def buildModel(self):
        self.model = Sequential()

        # Create input layer
        # Use Rectified Linear Unit (x > 0 ? 1 : 0)
        self.model.add(Dense(11, input_dim=11, activation='sigmoid'))
        self.model.add(Dense(11, activation='sigmoid'))
        self.model.add(Dense(11, activation='sigmoid'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='mean_squared_error',
                           optimizer='adam',
                           metrics=['accuracy'])

    def trainModel(self):
        self.model.fit(self.rTrainI, self.rTrainO, epochs=150, batch_size=400)
        _, accuracy = self.model.evaluate(self.rTrainI, self.rTrainO)
        print('Training Data Accuracy: %.2f' % (accuracy * 100))

    def testModel(self):
        _, accuracy = self.model.evaluate(self.rTestI, self.rTestO)
        print('Testing Data Accuracy: %.2f' % (accuracy * 100))

if __name__ == "__main__":
    wr = WineRater()
