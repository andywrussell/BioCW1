# We can use this file to run our program once it is done.

# For now I run some experiments flattening the net.
from ANN.neuralNet import NeuralNet
from ANN.layer import Layer
from PSO import PSO
import numpy as np
import pandas as pd
import os
from helpers import MSE
from helpers import read_data
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
max_runs = 1000


current_dir = os.getcwd() + '/'
inputs, ideal = read_data(current_dir, "2in_complex.txt")

my_pso = PSO(swarmsize, alpha, beta, gamma, delta, jumpsize, ideal, inputs, num_informants, max_runs, boundary)
my_pso.run_algo()

best_fitness = my_pso.best.outputs
print(ideal)
print(best_fitness)