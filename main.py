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
inputs, outputs = read_data(current_dir, "2in_complex.txt")


# Create a toy network
layer1 = Layer(input_count=2 , node_count=4, activations=[0,1,2,2])
layer1.build_layer()

layer2 = Layer(input_count=4 , node_count=1, activations=[2])
layer2.build_layer()

layers = [layer1, layer2]

# Check that neural net works
my_test_input = np.array([1,3])

network = NeuralNet(layers, my_test_input)
#network.fire_net()

# Explore the nets shape
#network.print_net()

# Result of neural net as a 1d vector
network.flatten_net()

# Change the values of the flatten net
network.net_as_vector = [i-5 for i in network.net_as_vector]

# Recovering the network
network.unflatten_net()

# Print the new values
#network.print_net()

# Flatten the net again (for tests)
network.flatten_net()
network.unflatten_net()
network.flatten_net()
network.unflatten_net()