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
import os

params_pso = {
    "swarmsize": 100,
    "alpha": 1,
    "beta": 1,
    "gamma": 1,
    "delta": 1,
    "jumpsize": 0.5,
    "boundary": 5,
    "num_informants": 10,
    "max_runs": 500
}

net_layers = {
    "layer1": {
        "input_count":1,
        "node_count":4,
        "activations": []
    },
    "layer2": {
        "input_count":4,
        "node_count": 1,
        "activations:":[]
    }
}

experiment1 = Experiment(params_pso, net_layers, path="1in_linear.txt", debugMode=True)
experiment1.run()