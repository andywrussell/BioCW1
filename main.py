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

#baseline.run_baseline()
#swarmsize.run_swarmsize()
alpha.run_alpha()

"""
params_pso = {
    "swarmsize": 50,
    "alpha": 1,
    "beta": 2.6,
    "gamma": 1.5,
    "delta": 0,
    "jumpsize": 1,
    "act_bound": 5,
    "weight_bound": 20,
    "bound_strat": 3,
    "num_informants": 5,
    "vel_range": 1,
    "max_runs": 1000,
    "informants_strat": 2
}

net_layers = {
    "layer1": {
        "input_count":1,
        "node_count":2,
        "activations": []
    },
    "layer2": {
        "input_count":2,
        "node_count": 2,
        "activations:":[]
    },
    "layer3": {
        "input_count":2,
        "node_count": 1,
        "activations:":[]
    }
}

experiment1 = Experiment(params_pso, net_layers, path="1in_tanh.txt", debugMode=False, sampleMode=True)
experiment1.run()

print(experiment1.pso.particles[0].best_list)
print(experiment1.pso.particles[0].best_list[-1])

<<<<<<< HEAD
experiment1 = Experiment(params_pso, net_complex, path="2in_xor.txt", debugMode=False, sampleMode=True)
=======
<<<<<<< HEAD
for particle in experiment1.pso.particles[0].informants:
    print(particle.best_list)
    print(particle.best_list[-2])
=======
experiment1 = Experiment(params_pso, net_layers, path="1in_linear.txt", debugMode=False, sampleMode=True)
>>>>>>> 1878f02a34cf2213115771c787f26aa8ae7ac8cb
experiment1.run()
"""
>>>>>>> 91a26e42f4d6214626219bf766234c88e5285d3d
