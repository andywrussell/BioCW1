# We can use this file to run our program once it is done.

# For now I run some experiments flattening the net.
from ANN.neuralNet import NeuralNet
from ANN.layer import Layer
from ANN.networkGenerator import NetworkGenerator
from PSO.pso import PSO
import numpy as np
import pandas as pd
import os
from utils.helpers import MSE, read_data
import os


swarmsize = 100
alpha = 1
beta = 1
gamma = 1
delta = 1
jumpsize = 0.5
#ideal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
boundary = 5
num_informants = 10
max_runs = 2

# Shape of our neural net
net_generator = NetworkGenerator()
net_generator.add_layer(input_count=2 , node_count=4)
net_generator.add_layer(input_count=4 , node_count=1)

current_dir = os.getcwd() + '/'
inputs, ideal = read_data(current_dir, "2in_complex.txt")

my_pso = PSO(net_generator, swarmsize, alpha, beta, gamma, delta, jumpsize, ideal, inputs, num_informants, max_runs, boundary)
my_pso.run_algo()

best_fitness = my_pso.best.outputs
print(my_pso.best.best_fitness)