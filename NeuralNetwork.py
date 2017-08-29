from NeuronLayer import NeuronLayer


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.nInputs = 0

    def setNumberOfInputs(self, n):
        self.nInputs = n

    def getInputLayer(self):
        return self.layers[0]

    def getOutputLayer(self):
        return self.layers[-1]

    def addLayer(self, n):
        nInputs = self.nInputs
        prev = None
        if len(self.layers) > 0:
            prev = self.getOutputLayer()
            nInputs = prev.getN()
        self.layers.append(NeuronLayer(n, nInputs, prev))

    def feed(self, input):
        return self.getInputLayer().feed(input)

    def errorBackpropagation(self, desOut):
        self.getOutputLayer().errorBackpropagation(desOut)

    def update(self, input):
        self.getInputLayer().update(input)

    def train(self, input, desOut):
        output = self.feed(input)
        self.errorBackpropagation(desOut)
        self.update(input)
        return output