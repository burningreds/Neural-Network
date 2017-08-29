import random
import matplotlib.pyplot as plt
import numpy as np


# Perceptron class
# Default weights and bias
class Neuron:
    def __init__(self, nInputs):
        self.weights = [(random.uniform(0, 1)) for i in range(0, nInputs)]
        self.learningRate = 0.5
        self.bias = 0.5
        self.error = 0
        self.delta = 0
        self.output = 0

    # Calculates result for inputs x1 and x2
    # x1 and x2 are bits
    def operate(self, inputs):
        self.output = self.sigmoidFunction(inputs)
        #return 1 if self.output > 0 else 0
        return self.output

    def sigmoidFunction(self, inputs):
        return 1 / (1 + np.exp(- sum([a*b for a,b in zip(self.weights, inputs)])))

    def setDelta(self, error):
        self.delta = error * self.output * (1.0 - self.output)

    def setWeights(self, input):
        newWeights = map(lambda i : self.weights[i] + self.learningRate * self.delta * input[i],
                         range(0, len(self.weights)))
        self.weights = newWeights

    def update(self, input):
        self.setWeights(input)
        self.bias = self.bias + self.learningRate * self.delta

    def getWeight(self, i):
        return self.weights[i]

    def getDelta(self):
        return self.delta

    def getOutput(self):
        return self.output