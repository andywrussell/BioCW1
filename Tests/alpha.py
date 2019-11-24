from ANN.neuralNet import NeuralNet
from ANN.layer import Layer
from ANN.networkGenerator import NetworkGenerator
from experiments import Experiment
from PSO.pso import PSO
import numpy as np
import pandas as pd
from utils.helpers import MSE, read_data
from Tests.networks.net_architectures import net_simple, net_complex_1, net_complex_2
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

def run_alpha():
    print("\nSwarmsize Alpha")
    print("=======================")

    params_pso = {
        "swarmsize": 40,
        "alpha": 0,
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

    cubic_optimal_alpha = 0
    cubic_best = None
    for j in range(0, 10):
        print("\nRun ", j)
        cubic_optimal_size = 0
        cubic_best = None
        for i in range(0, 10):      
            params_pso["alpha"] += 10
            experiment1 = Experiment(params_pso, net_simple, path="1in_cubic.txt", debugMode=False, sampleMode=True)
            experiment1.run()
            if (cubic_best == None or experiment1.pso.best.fitness < cubic_best):
                cubic_best = experiment1.pso.best.fitness
                cubic_optimal_alpha = params_pso["alpha"]
        print("\nRun ", j, "best alpha", cubic_optimal_alpha, " produced", cubic_best)

    print("cubic optimal alpha ", cubic_optimal_alpha, " produced", cubic_best)

    print("\nSwarmsize Linear")
    print("=======================")
    linear_optimal_alpha = 0
    linear_best = None
    for j in range(0, 10):
        print("\nRun ", j)
        linear_optimal_size = 0
        linear_best = None
        for i in range(0, 10):      
            params_pso["alpha"] += 10
            experiment1 = Experiment(params_pso, net_simple, path="1in_linear.txt", debugMode=False, sampleMode=True)
            experiment1.run()
            if (linear_best == None or experiment1.pso.best.fitness < linear_best):
                linear_best = experiment1.pso.best.fitness
                linear_optimal_alpha = params_pso["alpha"]
        print("\nRun ", j, "best alpha", linear_optimal_alpha, " produced", linear_best)

    print("linear optimal alpha ", linear_optimal_alpha, " produced", linear_best)

    print("\nSwarmsize Sine")
    print("=======================")    
    sine_optimal_alpha = 0
    sine_best = None
    for j in range(0, 10):
        print("\nRun ", j)
        sine_optimal_size = 0
        sine_best = None
        for i in range(0, 10):      
            params_pso["alpha"] += 10
            experiment1 = Experiment(params_pso, net_simple, path="1in_sine.txt", debugMode=False, sampleMode=True)
            experiment1.run()
            if (sine_best == None or experiment1.pso.best.fitness < sine_best):
                sine_best = experiment1.pso.best.fitness
                sine_optimal_alpha = params_pso["alpha"]
        print("\nRun ", j, "best alpha", sine_optimal_alpha, " produced", sine_best)

    print("sine optimal alpha ", sine_optimal_alpha, " produced", sine_best)

    print("\nSwarmsize Tanh")
    print("=======================")
    tanh_optimal_alpha = 0
    tanh_best = None
    for j in range(0, 10):
        print("\nRun ", j)
        tanh_optimal_size = 0
        tanh_best = None
        for i in range(0, 10):      
            params_pso["alpha"] += 10
            experiment1 = Experiment(params_pso, net_simple, path="1in_tanh.txt", debugMode=False, sampleMode=True)
            experiment1.run()
            if (tanh_best == None or experiment1.pso.best.fitness < tanh_best):
                tanh_best = experiment1.pso.best.fitness
                tanh_optimal_alpha = params_pso["alpha"]
        print("\nRun ", j, "best alpha", tanh_optimal_alpha, " produced", tanh_best)

    print("tanh optimal alpha ", tanh_optimal_alpha, " produced", tanh_best)


    print("\nSwarmsize Complex")
    print("=======================")
    complex_optimal_alpha = 0
    complex_best = None
    for j in range(0, 10):
        print("\nRun ", j)
        complex_optimal_size = 0
        complex_best = None
        for i in range(0, 10):      
            params_pso["alpha"] += 10
            experiment1 = Experiment(params_pso, net_complex_1, path="2in_complex.txt", debugMode=False, sampleMode=True)
            experiment1.run()
            if (complex_best == None or experiment1.pso.best.fitness < complex_best):
                complex_best = experiment1.pso.best.fitness
                complex_optimal_alpha = params_pso["alpha"]
        print("\nRun ", j, "best alpha", complex_optimal_alpha, " produced", complex_best)

    print("complex optimal alpha ", complex_optimal_alpha, " produced", complex_best)

    print("\nSwarmsize XOR")
    print("=======================")
    xor_optimal_size = 0
    xor_best = None
    for j in range(0, 10):
        print("\nRun ", j)
        xor_optimal_size = 0
        xor_best = None
        for i in range(0, 10):      
            params_pso["swarmsize"] += 10
            experiment1 = Experiment(params_pso, net_complex_2, path="2in_xor.txt", debugMode=False, sampleMode=True)
            experiment1.run()
            if (xor_best == None or experiment1.pso.best.fitness < xor_best):
                xor_best = experiment1.pso.best.fitness
                xor_optimal_size = params_pso["swarmsize"]
            print("\nRun ", j, "best size", xor_optimal_size, " produced", xor_best)

    print("xor optimal size ", xor_optimal_size, " produced", xor_best)
