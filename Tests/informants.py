##First find optimal informants strategy
##Use linear as a baseline since it speeds up the experiments

from ANN.neuralNet import NeuralNet
from ANN.layer import Layer
from ANN.networkGenerator import NetworkGenerator
from experiments import Experiment
from PSO.pso import PSO
import numpy as np
import pandas as pd
from utils.helpers import MSE, read_data
import os


def run_informant_strat():

    params_pso = {
        "swarmsize": 40,
        "alpha": 1,
        "beta": 1.18,
        "gamma": 2.92,
        "delta": 0,
        "jumpsize": 1,
        "act_bound": 5,
        "weight_bound": 10,
        "bound_strat": 1,
        "num_informants": 3,
        "vel_range": 1,
        "max_runs": 1000,
        "informants_strat": 0
    } 

    net_simple = {
        "layer1": {
            "input_count":1,
            "node_count":1,
            "activations": []
        }
    }

    net_simple_2 = {
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

    exp2 = 0
    for i in range(0, 10):
        print("\nRun ", i)
        experiment2 = Experiment(params_pso, net_simple, path="1in_sine.txt", debugMode=False, sampleMode=True)
        experiment2.run()
        exp2 += experiment2.pso.best.fitness

    print("\nMse for strat 0 on linear", exp2/10)

    exp2 = 0
    params_pso["informants_strat"] = 1
    for i in range(0, 10):
        print("\nRun ", i)
        experiment2 = Experiment(params_pso, net_simple, path="1in_sine.txt", debugMode=False, sampleMode=True)
        experiment2.run()
        exp2 += experiment2.pso.best.fitness

    print("\nMse for strat 1 on linear", exp2/10)

    exp2 = 0
    params_pso["informants_strat"] = 2
    for i in range(0, 10):
        print("\nRun ", i)
        experiment2 = Experiment(params_pso, net_simple, path="1in_sine.txt", debugMode=False, sampleMode=True)
        experiment2.run()
        exp2 += experiment2.pso.best.fitness

    print("\nMse for strat 2 on linear", exp2/10)

    exp2 = 0
    params_pso["informants_strat"] = 3
    for i in range(0, 10):
        print("\nRun ", i)
        experiment2 = Experiment(params_pso, net_simple, path="1in_sine.txt", debugMode=False, sampleMode=True)
        experiment2.run()
        exp2 += experiment2.pso.best.fitness

    print("\nMse for strat 2 on linear", exp2/10) 

def run_informant_count():

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
        "num_informants": 1,
        "vel_range": 1,
        "max_runs": 1000,
        "informants_strat": 0
    } 

    net_simple = {
        "layer1": {
            "input_count":1,
            "node_count":1,
            "activations": []
        }
    }

    for i in range(40):
        params_pso["num_informants"] = 1
        print("\nRun ", i)
        experiment2 = Experiment(params_pso, net_simple, path="1in_sine.txt", debugMode=False, sampleMode=True)
        experiment2.run()
        print("Swarm Size", i, "Results : ", experiment2.pso.best.fitness)

