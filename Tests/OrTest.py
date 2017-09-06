import unittest
from NeuralNetwork import *


# Tests for all classes
# Run to excecute all tests

# Or tests
class OrTests(unittest.TestCase):
    def setUp(self):
        self.neuralNetwork = NeuralNetwork()
        self.neuralNetwork.setNumberOfInputs(2)
        self.neuralNetwork.addLayer(3)
        self.neuralNetwork.addLayer(1)
        inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
        desOuts = map(lambda x: [x[0] or x[1]], inputs)
        self.neuralNetwork.training(inputs, desOuts, 5000)

    def testA(self):
        assert self.neuralNetwork.feed([1,1])[0] > 0.9, "Or not calculating values correctly"

    def testB(self):
        assert self.neuralNetwork.feed([1,0])[0] > 0.9, "Or not calculating values correctly"

    def testC(self):
        assert self.neuralNetwork.feed([0,1])[0] > 0.9, "Or not calculating values correctly"

    def testD(self):
        assert self.neuralNetwork.feed([0,0])[0] < 0.1, "Or not calculating values correctly"

if __name__ == '__main__':
    unittest.main(exit=False)