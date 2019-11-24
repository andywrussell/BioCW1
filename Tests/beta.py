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

def run_beta():
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

    net_layers = {
        "layer1": {
            "input_count":1,
            "node_count":1,
            "activations": []
        }
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

    best_gamma = 0
    best_beta = 0
    best_error = None

    for j in range(0, 10):
        run_beta = 0
        run_gamma = 4.1
        run_best = None

        #first do 4.1 and 0
        params_pso["beta"] = 0
        params_pso["gamma"] = 4.1

        experiment1 = Experiment(params_pso, net_layers, path="1in_sine.txt", debugMode=False, sampleMode=True)
        experiment1.run()

        if (run_best == None or experiment1.pso.best.fitness < run_best):
            run_best = experiment1.pso.best.fitness
            run_beta = params_pso["beta"]
            run_gamma = params_pso["gamma"]

        #first do 0.5 and 3.6
        params_pso["beta"] = 0.5
        params_pso["gamma"] = 3.6

        experiment1 = Experiment(params_pso, net_layers, path="1in_sine.txt", debugMode=False, sampleMode=True)
        experiment1.run()

        if (run_best == None or experiment1.pso.best.fitness < run_best):
            run_best = experiment1.pso.best.fitness
            run_beta = params_pso["beta"]
            run_gamma = params_pso["gamma"]

        #first do 1 and 3.1
        params_pso["beta"] = 1.0
        params_pso["gamma"] = 3.1

        experiment1 = Experiment(params_pso, net_layers, path="1in_sine.txt", debugMode=False, sampleMode=True)
        experiment1.run()

        if (run_best == None or experiment1.pso.best.fitness < run_best):
            run_best = experiment1.pso.best.fitness
            run_beta = params_pso["beta"]
            run_gamma = params_pso["gamma"]


        #first do 1 and 3.1
        params_pso["beta"] = 1.5
        params_pso["gamma"] = 2.6

        experiment1 = Experiment(params_pso, net_layers, path="1in_sine.txt", debugMode=False, sampleMode=True)
        experiment1.run()

        if (run_best == None or experiment1.pso.best.fitness < run_best):
            run_best = experiment1.pso.best.fitness
            run_beta = params_pso["beta"]
            run_gamma = params_pso["gamma"]


        #first do 1 and 3.1
        params_pso["beta"] = 2.05
        params_pso["gamma"] = 2.05

        experiment1 = Experiment(params_pso, net_layers, path="1in_sine.txt", debugMode=False, sampleMode=True)
        experiment1.run()

        if (run_best == None or experiment1.pso.best.fitness < run_best):
            run_best = experiment1.pso.best.fitness
            run_beta = params_pso["beta"]
            run_gamma = params_pso["gamma"]

        #first do 1 and 3.1
        params_pso["beta"] = 2.6
        params_pso["gamma"] = 1.5

        experiment1 = Experiment(params_pso, net_layers, path="1in_sine.txt", debugMode=False, sampleMode=True)
        experiment1.run()

        if (run_best == None or experiment1.pso.best.fitness < run_best):
            run_best = experiment1.pso.best.fitness
            run_beta = params_pso["beta"]
            run_gamma = params_pso["gamma"]

        #first do 1 and 3.1
        params_pso["beta"] =  3.1
        params_pso["gamma"] = 1.0

        experiment1 = Experiment(params_pso, net_layers, path="1in_sine.txt", debugMode=False, sampleMode=True)
        experiment1.run()

        if (run_best == None or experiment1.pso.best.fitness < run_best):
            run_best = experiment1.pso.best.fitness
            run_beta = params_pso["beta"]
            run_gamma = params_pso["gamma"]


        #first do 1 and 3.1
        params_pso["beta"] =  0.5
        params_pso["gamma"] = 3.6

        experiment1 = Experiment(params_pso, net_layers, path="1in_sine.txt", debugMode=False, sampleMode=True)
        experiment1.run()

        if (run_best == None or experiment1.pso.best.fitness < run_best):
            run_best = experiment1.pso.best.fitness
            run_beta = params_pso["beta"]
            run_gamma = params_pso["gamma"]


        #first do 1 and 3.1
        params_pso["beta"] =  0.0
        params_pso["gamma"] = 4.1

        experiment1 = Experiment(params_pso, net_layers, path="1in_sine.txt", debugMode=False, sampleMode=True)
        experiment1.run()

        if (run_best == None or experiment1.pso.best.fitness < run_best):
            run_best = experiment1.pso.best.fitness
            run_beta = params_pso["beta"]
            run_gamma = params_pso["gamma"]

        print("\nRun ", j, " Beta: ", run_beta, " Gamma: ", run_gamma, " Error", run_best)

    print("\nOverall Beta: ", best_beta, " Gamma: ", best_gamma, " Error", best_error)
