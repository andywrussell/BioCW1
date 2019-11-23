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
from Tests import baseline
import os

print("\nExperiment 1")
print("=======================")

params_pso = {
    "swarmsize": 50,
    "alpha": 1,
    "beta": 1,
    "gamma": 1,
    "delta": 0,
    "jumpsize": 1,
    "act_bound": 5,
    "weight_bound": 0,
    "num_informants": 5,
    "max_runs": 100
}

net_layers = {
    "layer1": {
        "input_count":1,
        "node_count":2,
        "activations": []
    },
    "layer2": {
        "input_count":2,
        "node_count": 1,
        "activations:":[]
    }
}


experiment1 = Experiment(params_pso, net_layers, path="1in_linear.txt", debugMode=True)
experiment1.run()


print("\nExperiment 2")
print("=======================")

params_pso = {
    "swarmsize": 50,
    "alpha": 0.5,
    "beta": 0.5,
    "gamma": 0.5,
    "delta": 0.5,
    "jumpsize": 0.5,
    "act_bound": 5,
    "weight_bound": 0.5,
    "num_informants": 50,
    "max_runs": 100
}

net_layers = {
    "layer1": {
        "input_count":1,
        "node_count":8,
        "activations": []
    },
    "layer2": {
        "input_count":8,
        "node_count": 1,
        "activations:":[]
    }
}

exp2 = Experiment(params_pso, net_layers, path="1in_linear.txt", debugMode=True)
exp2.run()


print("\nExperiment 3")
print("=======================")

params_pso = {
    "swarmsize": 100,
    "alpha": 0.5,
    "beta": 0.5,
    "gamma": 1,
    "delta": 0,
    "jumpsize": 0.9,
    "act_bound": 5,
    "weight_bound": 0,
    "num_informants": 50,
    "max_runs": 100
}

net_layers = {
    "layer1": {
        "input_count":1,
        "node_count":10,
        "activations": []
    },
    "layer2": {
        "input_count":10,
        "node_count": 1,
        "activations:":[]
    }
}


exp3 = Experiment(params_pso, net_layers, path="1in_linear.txt", debugMode=True)
exp3.run()

print("\nExperiment 4")
print("=======================")
params_pso = {
    "swarmsize": 50,
    "alpha": 1,
    "beta": 0,
    "gamma": 0,
    "delta": 0,
    "jumpsize": 0.9,
    "act_bound": 5,
    "weight_bound": 0,
    "num_informants": 5,
    "max_runs": 100
}

net_layers = {
    "layer1": {
        "input_count":1,
        "node_count":100,
        "activations": []
    },
    "layer2": {
        "input_count":100,
        "node_count": 1,
        "activations:":[]
    }
}

exp4 = Experiment(params_pso, net_layers, path="1in_linear.txt", debugMode=True)
exp4.run()

print("\nExperiment 5")
print("=======================")

params_pso = {
    "swarmsize": 150,
    "alpha": 0.2,
    "beta": 0.2,
    "gamma": 0.2,
    "delta": 0,
    "jumpsize": 0.1,
    "act_bound": 5,
    "weight_bound": 0,
    "num_informants": 10,
    "max_runs": 40
}

net_layers = {
    "layer1": {
        "input_count":1,
        "node_count":30,
        "activations": []
    },
    "layer2": {
        "input_count":30,
        "node_count": 8,
        "activations:":[]
    },
    "layer3": {
        "input_count":8,
        "node_count": 1,
        "activations:":[]
    }
}

exp5 = Experiment(params_pso, net_layers, path="1in_linear.txt", debugMode=False)
exp5.run()

print("\nExperiment 6")
print("=======================")
params_pso = {
    "swarmsize": 50,
    "alpha": 1,
    "beta": 1,
    "gamma": 1,
    "delta": 0,
    "jumpsize": 0.1,
    "act_bound": 5,
    "weight_bound": 0,
    "num_informants": 5,
    "max_runs": 100
}

net_layers = {
    "layer1": {
        "input_count":1,
        "node_count":5,
        "activations": []
    },
    "layer2": {
        "input_count":5,
        "node_count": 1,
        "activations:":[]
    }
}

exp6 = Experiment(params_pso, net_layers, path="1in_linear.txt", debugMode=True)
exp6.run()