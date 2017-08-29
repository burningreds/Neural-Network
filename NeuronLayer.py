from Neuron import Neuron


class NeuronLayer:
    def __init__(self, n, nInputs, prev):
        self.neurons = self.generateNeurons(n, nInputs)
        self.nextLayer = None
        self.previousLayer = prev
        self.nInputs = nInputs
        self.n = n
        if not self.previousLayer is None:
            self.previousLayer.setNextLayer(self)

    def generateNeurons(self, n, nInputs):
        neurons = []
        for i in range(0, n):
            neurons.append(Neuron(nInputs))
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

    def feed(self, input):
        myOut = map(lambda x: x.operate(input), self.getNeurons())
        if not self.isOutputLayer():
            return self.nextLayer.feed(myOut)
        return myOut

    def errorBackpropagation(self, desOut):
        if self.isOutputLayer():
            error = map(lambda n, o : o - n.output, self.getNeurons(), desOut)
        else:
            error = map (
                lambda i : sum( map (
                    lambda n : n.getWeight(i) * n.getDelta(),
                    self.nextLayer.getNeurons())),
                range(0, self.getN()))
        map(lambda e, n : n.setDelta(e), error, self.getNeurons())
        if not self.isInputLayer():
            self.previousLayer.errorBackpropagation(desOut)

    def update(self, input):
        map(lambda n : n.update(input), self.getNeurons())
        if not self.isOutputLayer():
            self.nextLayer.update(self.getOutputs())