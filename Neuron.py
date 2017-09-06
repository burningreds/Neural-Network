import random
import numpy as np


# Perceptron class
# Default weights and bias
class Neuron:
    def __init__(self, nInputs):
        self.weights = [(random.uniform(-1, 1)) for i in range(0, nInputs)]
        self.learningRate = 0.5
        self.bias = 0.5
        self.error = 0
        self.delta = 0
        self.output = 0

    def getWeight(self, i):
        return self.weights[i]

    def getDelta(self):
        return self.delta

    def getOutput(self):
        return self.output

    # Calculates result for inputs with sigmoid function
    def operate(self, inputs):
        self.output = self.sigmoidFunction(inputs)
        return self.output

    # Sigmoid Function
    def sigmoidFunction(self, inputs):
        return 1 / (1 + np.exp(- sum([a*b for a,b in zip(self.weights, inputs)]) - self.bias))

    def setDelta(self, error):
        self.delta = error * self.output * (1.0 - self.output)

    def setWeights(self, input):
        newWeights = map(lambda i : self.weights[i] + self.learningRate * self.delta * input[i],
                         range(0, len(self.weights)))
        self.weights = newWeights

    def update(self, input):
        self.setWeights(input)
        self.bias = self.bias + self.learningRate * self.delta