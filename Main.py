from NeuralNetwork import *
import random


def main():
    nn = NeuralNetwork()
    nn.setNumberOfInputs(2)
    nn.addLayer(2)
    nn.addLayer(1)
    training(nn)
    print(nn.feed([1, 1]))
    print(nn.feed([1, 0]))
    print(nn.feed([0, 1]))
    print(nn.feed([0, 0]))

def training(neuralNetwork):
    for i in range(0, 1000):
        input = [random.randint(0, 1), random.randint(0,1)]
        desOut = [input[0] and input[1]]
        neuralNetwork.train(input,desOut)

if __name__ == "__main__":
    main()