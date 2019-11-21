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

current_dir = os.getcwd() + '/'
inputs, ideal = read_data(current_dir, "2in_xor.txt")


params_pso = {
    "swarmsize": 60,
    "alpha": 1,
    "beta": 1,
    "gamma": 1,
    "delta": 1,
    "jumpsize": 0.5,
    "ideal": inputs,
    "inputs": ideal,
    "boundary": 5,
    "num_informants": 10,
    "max_runs": 30
}

net_layers = {
    "layer1": {
        "input_count":2,
        "node_count":4,
        "activations": []
    },
    "layer2": {
        "input_count":4,
        "node_count": 1,
        "activations:":[]
    }
}

experiment1 = Experiment(params_pso, net_layers, path="2in_xor.txt")
experiment1.run()