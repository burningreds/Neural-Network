from Neuron import Neuron


class NeuronLayer:
    def __init__(self, n, nInputs, prev, learningRate):
        self.neurons = self.generateNeurons(n, nInputs, learningRate)
        self.nextLayer = None
        self.previousLayer = prev
        self.nInputs = nInputs
        self.n = n #number of neurons
        if not self.previousLayer is None:
            self.previousLayer.setNextLayer(self)

    # Returns a list of n neurons
    # Each neuron needs to know how many inputs it receives
    def generateNeurons(self, n, nInputs, learningRate):
        neurons = []
        for i in range(0, n):
            neurons.append(Neuron(nInputs, learningRate))
        return neurons

    def isOutputLayer(self):
        return self.nextLayer is None

    def isInputLayer(self):
        return self.previousLayer is None

    def setNInputs(self, n):
        self.nInputs = n

    def setNextLayer(self, layer):
        self.nextLayer = layer

    def getNOutputs(self):
        return self.nextLayer.getN()

    def getNInputs(self):
        return self.nInputs

    def getNeurons(self):
        return self.neurons

    def getN(self):
        return self.n

    def getOutputs(self):
        return map(lambda n: n.getOutput(), self.getNeurons())

    # Feeds the neurons in the layer with the input
    def feed(self, input):
        myOut = map(lambda x: x.operate(input), self.getNeurons())
        # If it's not the last layer, continue with the next
        if not self.isOutputLayer():
            return self.nextLayer.feed(myOut)
        # If it's the last layer, return output
        return myOut

    # Backward-propagates the error
    # There are 2 cases: output layer or a hidden layer
    def errorBackpropagation(self, desOut):
        if self.isOutputLayer():
            error = map(lambda n, o : o - n.getOutput(), self.getNeurons(), desOut)
        else:
            error = map (
                lambda i : sum( map (
                    lambda n : n.getWeight(i) * n.getDelta(),
                    self.nextLayer.getNeurons())),
                range(0, self.getN()))
        map(lambda e, n : n.setDelta(e), error, self.getNeurons())
        if not self.isInputLayer():
            self.previousLayer.errorBackpropagation(desOut)

    # Updates weights and bias for every neuron in the layer
    def update(self, input):
        map(lambda n : n.update(input), self.getNeurons())
        if not self.isOutputLayer():
            self.nextLayer.update(self.getOutputs())