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

    # exp1 = 0
    # for i in range(0, 10):
    #     print("\nRun ", i)
    #     experiment1 = Experiment(params_pso, net_layers, path="1in_cubic.txt", debugMode=False, sampleMode=True)
    #     experiment1.run()
    #     exp1 += experiment1.pso.best.fitness

    # print("\nMse for base on cubic", exp1/10)

    net_layers = {
        "layer1": {
            "input_count":1,
            "node_count":1,
            "activations": []
        }
    }

    print("\nBase Linear")
    print("=======================")
    exp2 = 0
    for i in range(0, 10):
        print("\nRun ", i)
        experiment2 = Experiment(params_pso, net_layers, path="1in_linear.txt", debugMode=False, sampleMode=True)
        experiment2.run()
        exp2 += experiment2.pso.best.fitness

    print("\nMse for base on linear", exp2/10)


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

    print("\nBase Sine")
    print("=======================")
    exp3 = 0
    for i in range(0, 10):
        print("Run ", i, "\n")
        experiment3 = Experiment(params_pso, net_layers, path="1in_sine.txt", debugMode=False, sampleMode=True)
        experiment3.run()
        exp3 += experiment3.pso.best.fitness

    print("\nMse for base on Sine", exp3/10)

    print("\nBase Tanh")
    print("=======================")
    exp4 = 0
    for i in range(0, 10):
        print("Run ", i, "\n")
        experiment4 = Experiment(params_pso, net_layers, path="1in_tanh.txt", debugMode=False, sampleMode=True)
        experiment4.run()
        exp4 += experiment4.pso.best.fitness

    print("\nMse for base on Tanh", exp4/10)    


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

    print("\nBase Complex")
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
    
    exp5 = 0
    for i in range(0, 10):
        print("\nRun ", i, "\n")
        experiment5 = Experiment(params_pso, net_complex, path="2in_complex.txt", debugMode=False, sampleMode=True)
        experiment5.run()
        exp5 += experiment5.pso.best.fitness

    print("\nMse for base on Complex", exp5/10)   

    print("\nBase XOR")
    print("=======================")
    exp6 = 0
    for i in range(0, 10):
        print("Run ", i, "\n")
        experiment6 = Experiment(params_pso, net_complex, path="2in_xor.txt", debugMode=False, sampleMode=True)
        experiment6.run()
        exp6 += experiment6.pso.best.fitness

    print("\nMse for base on Complex", exp6/10)   
