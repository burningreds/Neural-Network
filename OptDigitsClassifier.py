import time
from NeuralNetwork import *

def main():
    neuralNetwork = NeuralNetwork(0.4)
    neuralNetwork.setNumberOfInputs(64)
    neuralNetwork.addLayer(35)
    neuralNetwork.addLayer(35)
    neuralNetwork.addLayer(10)
    training = open('Data/optdigits.tra', 'r')
    test = open('Data/optdigits.tes', 'r')
    start = time.time()
    print "training"
    trainClassifier(neuralNetwork, training, 5)
    print "testing"
    testClassifier(neuralNetwork, test)
    end = time.time()
    print "Tiempo de ejecucion: " + str(end - start)


def trainClassifier(neuralNetwork, training, epochs):
    inputs = []
    outputs = []
    for line in training:
        lineArray = line.split(',')
        inputs.append([int(x) / 16.0 for x in lineArray[0:-1]])
        output = [0]*10
        output[int(lineArray[-1])] = 1
        outputs.append(output)
    neuralNetwork.training(inputs, outputs, epochs)

def testClassifier(neuralNetwork, test):
    successCount = 0
    total = 0
    for line in test:
        lineArray = line.split(',')
        input = ([int(x) / 16.0 for x in lineArray[0:-1]])
        output = int(lineArray[-1])
        successCount += neuralNetwork.test(input, output)
        total += 1
    print "Tasa de exito: " + str(successCount * 1.0 / total)


if __name__ == "__main__":
    main()