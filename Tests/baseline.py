from ANN.neuralNet import NeuralNet
from ANN.layer import Layer
from ANN.networkGenerator import NetworkGenerator
from experiments import Experiment
from PSO.pso import PSO
import numpy as np
import pandas as pd
from utils.helpers import MSE, read_data
import os

#Swarmsize - Start with 50 - typical is 10 - 100 according to lecture slides
#alpha - start a 1
#beta, gamma, delta typically sum to 4
#acorrding to https://cs.gmu.edu/~sean/book/metaheuristics/Essentials.pdf delta is often set to 0
#explained on page 57
#gamma is often midpoint between delts and beta
#start with weight boundary between -1 and 1
#initially set informants to 3 according to page 5 of http://clerc.maurice.free.fr/pso/SPSO_descriptions.pdf

#Run the baseline expriment on each data set to see results

def run_baseline():
    print("\nBase Cubic")
    print("=======================")

    params_pso = {
        "swarmsize": 50,
        "alpha": 1,
        "beta": 2.67,
        "gamma": 1.33,
        "delta": 0,
        "jumpsize": 1,
        "act_bound": 5,
        "weight_bound": 10,
        "bound_strat": 3,
        "num_informants": 3,
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

    experiment1 = Experiment(params_pso, net_layers, path="1in_cubic.txt", debugMode=True)
    experiment1.run()


    print("\nBase Linear")
    print("=======================")

    exp2 = Experiment(params_pso, net_layers, path="1in_linear.txt", debugMode=True)
    exp2.run()


    print("\nBase Sine")
    print("=======================")


    exp3 = Experiment(params_pso, net_layers, path="1in_sine.txt", debugMode=True)
    exp3.run()

    print("\nBase Tanh")
    print("=======================")


    exp4 = Experiment(params_pso, net_layers, path="1in_tanh.txt", debugMode=True)
    exp4.run()

    print("\nBase Complex")
    print("=======================")

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

    exp5 = Experiment(params_pso, net_layers, path="2in_complex.txt", debugMode=False)
    exp5.run()

    print("\nBase XOR")
    print("=======================")

    exp6 = Experiment(params_pso, net_layers, path="2in_xor.txt", debugMode=True)
    exp6.run()