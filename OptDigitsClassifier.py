from NeuralNetwork import *

def main():
    neuralNetwork = NeuralNetwork()
    neuralNetwork.setNumberOfInputs(64)
    neuralNetwork.addLayer(35)
    neuralNetwork.addLayer(35)
    neuralNetwork.addLayer(10)

    training = open('Data/optdigits.tra', 'r')
    for line in training:
        lineArray = line.split(',')
        input = ([int(x) / 16.0 for x in lineArray[0:-1]])
        output = [0]*10
        output[int(lineArray[-1])] = 1
        neuralNetwork.train(input, output)

    successCount = 0
    total = 0
    test = open('Data/optdigits.tes', 'r')
    for line in test:
        lineArray = line.split(',')
        input = ([int(x) / 16.0 for x in lineArray[0:-1]])
        output = int(lineArray[-1])
        successCount += neuralNetwork.test(input, output)
        total += 1
    print successCount * 1.0 / total


if __name__ == "__main__":
    main()