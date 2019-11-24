# We can use this file to run our program once it is done.

# For now I run some experiments flattening the net.
from ANN.neuralNet import NeuralNet
from ANN.layer import Layer
from ANN.networkGenerator import NetworkGenerator
from experiments import Experiment
from PSO.pso import PSO
import numpy as np
import pandas as pd
from utils.helpers import MSE, read_data
from Tests import baseline, swarmsize, alpha
from Tests.networks.net_architectures import net_simple_1, net_simple_2, net_complex
import os

baseline.run_baseline()
#swarmsize.run_swarmsize()
#alpha.run_alpha()

# params_pso = {
#     "swarmsize": 40,
#     "alpha": 1,
#     "beta": 2.05,
#     "gamma": 2.05,
#     "delta": 0,
#     "jumpsize": 1,
#     "act_bound": 5,
#     "weight_bound": 20,
#     "bound_strat": 3,
#     "num_informants": 3,
#     "vel_range": 1,
#     "max_runs": 1000,
#     "informants_strat": 2
# }

# net_layers = {
#     "layer1": {
#         "input_count":1,
#         "node_count":3,
#         "activations": []
#     },
#     "layer2": {
#         "input_count":3,
#         "node_count": 5,
#         "activations:":[]
#     },
#     "layer3": {
#         "input_count":5,
#         "node_count": 1,
#         "activations:":[]
#     }    
# }

# experiment1 = Experiment(params_pso, net_layers, path="1in_sine.txt", debugMode=False, sampleMode=True)
# experiment1.run()
