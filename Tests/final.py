from ANN.neuralNet import NeuralNet
from ANN.layer import Layer
from ANN.networkGenerator import NetworkGenerator
from experiments import Experiment
from PSO.pso import PSO
import numpy as np
import pandas as pd
from utils.helpers import MSE, read_data
import os

def run_final():
    print("\Final Cubic")
    print("=======================")

    params_pso = {
        "swarmsize": 40,
        "alpha": 1,
        "beta": 0,
        "gamma": 4.1,
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

    net_single = {
        "layer1": {
            "input_count":1,
            "node_count":1,
            "activations": []
        }
    }

    exp1 = 0
    for i in range(0, 10):
        print("\nRun ", i)
        experiment1 = Experiment(params_pso, net_single, path="1in_cubic.txt", debugMode=False, sampleMode=True)
        experiment1.run()
        exp1 += experiment1.pso.best.fitness

    print("\nMse for final on cubic", exp1/10)


    params_pso["beta"] = 0.5
    params_pso["gamma"] = 3.6

    print("\Final Linear")
    print("=======================")
    exp2 = 0
    for i in range(0, 10):
        print("\nRun ", i)
        experiment2 = Experiment(params_pso, net_single, path="1in_linear.txt", debugMode=False, sampleMode=True)
        experiment2.run()
        exp2 += experiment2.pso.best.fitness

    print("\nMse for final on linear", exp2/10)

    params_pso["beta"] = 0
    params_pso["gamma"] = 4.1

    print("\Final Sine")
    print("=======================")
    exp3 = 0
    for i in range(0, 10):
        print("Run ", i, "\n")
        experiment3 = Experiment(params_pso, net_single, path="1in_sine.txt", debugMode=False, sampleMode=True)
        experiment3.run()
        exp3 += experiment3.pso.best.fitness

    print("\nMse for final on Sine", exp3/10)

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

    params_pso["beta"] = 0
    params_pso["gamma"] = 4.1

    print("\nFinal Tanh")
    print("=======================")
    exp4 = 0
    for i in range(0, 10):
        print("Run ", i, "\n")
        experiment4 = Experiment(params_pso, net_layers, path="1in_tanh.txt", debugMode=False, sampleMode=True)
        experiment4.run()
        exp4 += experiment4.pso.best.fitness

    print("\nMse for final on Tanh", exp4/10)    

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

    params_pso["beta"] = 0
    params_pso["gamma"] = 4.1

    print("\nFinal XOR")
    print("=======================")
    exp6 = 0
    for i in range(0, 10):
        print("\nRun ", i)
        experiment6 = Experiment(params_pso, net_layers, path="2in_xor.txt", debugMode=False, sampleMode=True)
        experiment6.run()
        exp6 += experiment6.pso.best.fitness

    print("\nMse for final on XOR", exp6/10)   

    print("\nFinal Complex")
    print("=======================")

    net_complex = {
        "layer1": {
            "input_count":2,
            "node_count":2,
            "activations": []
        },
        "layer2": {
            "input_count":2,
            "node_count":2 ,
            "activations:":[]
        },
        "layer3": {
            "input_count":2,
            "node_count":1 ,
            "activations:":[]
        }
    }

    params_pso["beta"] = 2.05
    params_pso["gamma"] = 2.05
    
    exp5 = 0
    for i in range(0, 10):
        print("\nRun ", i, "\n")
        experiment5 = Experiment(params_pso, net_complex, path="2in_complex.txt", debugMode=False, sampleMode=True)
        experiment5.run()
        exp5 += experiment5.pso.best.fitness

    print("\nMse for final on Complex", exp5/10)   


