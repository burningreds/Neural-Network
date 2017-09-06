from NeuronLayer import NeuronLayer


class NeuralNetwork:
    def __init__(self, learningRate):
        self.layers = []
        self.nInputs = 0
        self.learningRate = learningRate

    def setNumberOfInputs(self, n):
        self.nInputs = n

    def getInputLayer(self):
        return self.layers[0]

    def getOutputLayer(self):
        return self.layers[-1]

    # Adds a neuron layer with n neurons to the network
    def addLayer(self, n):
        nInputs = self.nInputs
        prev = None
        if len(self.layers) > 0:
            prev = self.getOutputLayer()
            nInputs = prev.getN()
        self.layers.append(NeuronLayer(n, nInputs, prev, self.learningRate))

    # Operation that consists in providing a set of inputs to
    # the network, and obtain a set of outputs
    def feed(self, input):
        return self.getInputLayer().feed(input)

    # Deltas and errors of every neuron are propagated
    # starting from the last (or output) layer
    def errorBackpropagation(self, desOut):
        self.getOutputLayer().errorBackpropagation(desOut)

    # Weights and biases are updated starting from the first layer
    def update(self, input):
        self.getInputLayer().update(input)

    # The complete training process
    # Forward feeding, backward propagating the errors
    # and updating neuron weights and biases
    def train(self, input, desOut):
        output = self.feed(input)
        self.errorBackpropagation(desOut)
        self.update(input)

    # Trains the nn for every input in training set
    # a certain amount of times (epochs)
    def training(self, inputs, desOuts, epochs):
        for i in range(0, epochs):
            for j in range(0, len(inputs)):
                self.train(inputs[j], desOuts[j])
        return

    #
    def test(self, input, desOut):
        output = self.feed(input)
        actualOutput = max(xrange(len(output)), key=lambda x: output[x])
        if desOut == actualOutput:
            return 1
        return 0