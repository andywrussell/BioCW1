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
from Tests import baseline, swarmsize, alpha, beta, informants, final
from Tests.networks.net_architectures import net_simple_1, net_simple_2, net_complex
import os


#Run this to see one iteration of base experiment for xor
#Output shows best fitness, number of iterations and the expected outputs vs the real outputs
params_pso = {
    "swarmsize": 40,
    "alpha": 1,
    "beta": 2.05,
    "gamma": 2.05,
    "delta": 0,
    "jumpsize": 1,
    "act_bound": 5,
    "weight_bound": 10,
    "bound_strat": 1,
    "num_informants": 3,
    "vel_range": 1,
    "max_runs": 1000,
    "informants_strat": 2
} 

net_layers = {
    "layer1": {
        "input_count":2,
        "node_count":2,
        "activations": []
    },
    "layer2": {
        "input_count":2,
        "node_count": 1,
        "activations:":[]
    }
}

xorexp = Experiment(params_pso, net_layers, path="2in_xor.txt", debugMode=False, sampleMode=True)
xorexp.run()

##Single iterations of base experiments commented below
# net_single = {
#     "layer1": {
#         "input_count":1,
#         "node_count":1,
#         "activations": []
#     }
# }

# cubicexp = Experiment(params_pso, net_single, path="1in_cubic.txt", debugMode=False, sampleMode=True)
# cubicexp.run()

# linearexp = Experiment(params_pso, net_single, path="1in_linear.txt", debugMode=False, sampleMode=True)
# linearexp.run()

# sinexp = Experiment(params_pso, net_single, path="1in_sine.txt", debugMode=False, sampleMode=True)
# sinexp.run()

# net_layers = {
#     "layer1": {
#         "input_count":1,
#         "node_count":2,
#         "activations": []
#     },
#     "layer2": {
#         "input_count":2,
#         "node_count": 1,
#         "activations:":[]
#     }
# }

# tanhexp = Experiment(params_pso, net_layers, path="1in_tanh.txt", debugMode=False, sampleMode=True)
# tanhexp.run()

# net_complex = {
#     "layer1": {
#         "input_count":2,
#         "node_count":2,
#         "activations": []
#     },
#     "layer2": {
#         "input_count":2,
#         "node_count":2 ,
#         "activations:":[]
#     },
#     "layer3": {
#         "input_count":2,
#         "node_count":1 ,
#         "activations:":[]
#     }
# }

# complexexp = Experiment(params_pso, net_complex, path="2in_complex.txt", debugMode=False, sampleMode=True)
# complexexp.run()