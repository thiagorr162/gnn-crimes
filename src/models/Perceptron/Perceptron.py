import numpy as np
import pandas as pd
class Perceptron:
    def __init__(self, featuresNumber, learningRate):
        #weights are n+1 due to bias neuron
        self.featuresNumber = featuresNumber
        self.weights = np.random.rand(featuresNumber + 1)
        self.learningRate = learningRate
        self.epochs = 0
    #activation function
    def compute(self, variablesArray):

        result = np.dot(self.weights, variablesArray)
        if(result < 0):
            return -1
        return 1
    #using stochastic gradient descent
    def train(self, trainingDataframe, epochLimit=0):
        while True:
            stop = 1
            for i in range(0, len(trainingDataframe)):
                row = trainingDataframe.iloc[i]
                variables = np.array(row[0:self.featuresNumber])
                #bias
                variables = np.append(variables, 1)
                guess = self.compute(variables)
                if(guess != row[-1]):
                    stop = 0
                    self.weights += self.learningRate * (row[-1] - guess) * variables
                self.epochs+=1
                if self.epochs == epochLimit:
                    stop = 1
                    break

            #has reachead convergence or epoch limit
            if(stop):
              break
    #returns a guess based on parameters
    def guess(self, variablesArray):
       variablesArray = np.append(variablesArray, 1)
       return self.compute(variablesArray)
    def test(self, testDataframe):
        errors = 0
        for i in range(0, len(testDataframe)):
                row = testDataframe.iloc[i]
                variables = np.array(row[0:self.featuresNumber])
                #bias
                variables = np.append(variables, 1)
                guess = self.compute(variables)
                if(guess != row[-1]):
                    errors += 1

        total = len(testDataframe)
        return (total - errors) / total

