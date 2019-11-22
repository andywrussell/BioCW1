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
    "swarmsize": 60,
    "alpha": 0.5,
    "beta": 0.5,
    "gamma": 0.5,
    "delta": 0,
    "jumpsize": 0.9,
    "boundary": 10,
    "num_informants": 10,
    "max_runs": 200
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

#experiment1 = Experiment(params_pso, net_layers, path="1in_linear.txt", debugMode=True)
#experiment1.run()

network = NeuralNet(MSE)
network.add_layer(1,4)
network.add_layer(4,1)
network.flatten_net()
network.unflatten_net()