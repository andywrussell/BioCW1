# We can use this file to run our program once it is done.

# For now I run some experiments flattening the net.
from ANN.neuralNet import NeuralNet
from ANN.layer import Layer
import numpy as np
import pandas as pd
import os
from helpers import MSE
from helpers import read_data
import os


current_dir = os.getcwd() + '/'
inputs, outputs = read_data(current_dir, "2in_xor.txt")

# Create a toy network
layer1 = Layer(input_count=2 , node_count=4)
layer1.build_layer()

layer2 = Layer(input_count=4 , node_count=1, activations=[3])
layer2.build_layer()

layers = [layer1, layer2]

# Check that neural net works
network = NeuralNet(layers, error_function=MSE)
fitness, predictions = network.get_fitness(inputs, outputs)

print(fitness)