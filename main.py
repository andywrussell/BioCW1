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
from Tests import baseline, swarmsize, alpha, beta, informants
from Tests.networks.net_architectures import net_simple_1, net_simple_2, net_complex
import os

#beta.run_beta()

informants.run_informant_strat()

#baseline.run_baseline()
#swarmsize.run_swarmsize()
#alpha.run_alpha()

# params_pso = {
#     "swarmsize": 40,
#     "alpha": 1,
#     "beta": 1.18,
#     "gamma": 2.92,
#     "delta": 0,
#     "jumpsize": 1,
#     "act_bound": 5,
#     "weight_bound": 10,
#     "bound_strat": 1,
#     "num_informants": 3,
#     "vel_range": 1,
#     "max_runs": 1000,
#     "informants_strat": 2
# }
# net_single = {
#     "layer1": {
#         "input_count":1,
#         "node_count":1,
#         "activations": []
#     }
# }

# experiment1 = Experiment(params_pso, net_single, path="1in_cubic.txt", debugMode=False, sampleMode=True)
# experiment1.run()
