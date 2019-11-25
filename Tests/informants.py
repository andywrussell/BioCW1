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
        "beta": 0.805,
        "gamma":  3.295,
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

    results = open("informantsresults.txt","w+")
    results.write("\n\nTEST")

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
        "num_informants": 0,
        "vel_range": 1,
        "max_runs": 1000,
        "informants_strat": 2
    } 

    net_simple = {
        "layer1": {
            "input_count":1,
            "node_count":1,
            "activations": []
        }
    }

    params_pso["beta"] = 1.18
    params_pso["gamma"] = 2.92

    bestError = None
    bestNum = 0
    params_pso["num_informants"] = 0
    for i in range(40):
        params_pso["num_informants"] = i+1
        error = 0 
        for j in range(10):
          #  print("\nRun ", i)
            experiment2 = Experiment(params_pso, net_simple, path="1in_linear.txt", debugMode=False, sampleMode=True)
            experiment2.run()
            error += experiment2.pso.best.fitness
        avg = error/10
        results.write("\nLinear - Avg error for " + str(i) + " informants: " +  str(avg))
        if (bestError == None or avg < bestError):
            bestError = avg
            bestNum = i+1
    results.write("\nBest Informants for linear " + str(bestNum) + " Informants : " + str(bestError))

    #Cubic
    params_pso["beta"] = 1.325
    params_pso["gamma"] = 2.775

    bestError = None
    bestNum = 0
    params_pso["num_informants"] = 0
    for i in range(40):
        params_pso["num_informants"] = i+1
        error = 0 
        for j in range(10):
          #  print("\nRun ", i)
            experiment2 = Experiment(params_pso, net_simple, path="1in_cubic.txt", debugMode=False, sampleMode=True)
            experiment2.run()
            error += experiment2.pso.best.fitness
        avg = error/10
        results.write("\nCubic - Avg error for " + str(i) + " informants: " + str(avg))
        if (bestError == None or avg < bestError):
            bestError = avg
            bestNum = i+1
    results.write("\nBest Informants for cubic " + str(bestNum) + " Informants : " + str(bestError))

    #Sine
    params_pso["beta"] = 0.91
    params_pso["gamma"] = 3.19

    bestError = None
    bestNum = 0
    params_pso["num_informants"] = 0
    for i in range(40):
        params_pso["num_informants"] = i+1
        error = 0 
        for j in range(10):
          #  print("\nRun ", i)
            experiment2 = Experiment(params_pso, net_simple, path="1in_sine.txt", debugMode=False, sampleMode=True)
            experiment2.run()
            error += experiment2.pso.best.fitness
        avg = error/10
        results.write("\nSine - Avg error for " + str(i) + " informants: " + str(avg))
        if (bestError == None or avg < bestError):
            bestError = avg
            bestNum = i+1
    results.write("\nBest Informants for sine " + str(bestNum) + " Informants : " + str(bestError)) 

    #Tanh
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

    params_pso["beta"] = 0.805
    params_pso["gamma"] = 3.295

    bestError = None
    bestNum = 0
    params_pso["num_informants"] = 0
    for i in range(40):
        params_pso["num_informants"] = i+1
        error = 0 
        for j in range(10):
          #  print("\nRun ", i)
            experiment2 = Experiment(params_pso, net_layers, path="1in_tanh.txt", debugMode=False, sampleMode=True)
            experiment2.run()
            error += experiment2.pso.best.fitness
        avg = error/10
        results.write("\nTanh - Avg error for " + str(i) + " informants: " + str(avg))
        if (bestError == None or avg < bestError):
            bestError = avg
            bestNum = i+1
    results.write("\nBest Informants for tanh " + str(bestNum) + " Informants : " + str(bestError)) 

    #xor
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

    params_pso["beta"] = 1.125
    params_pso["gamma"] = 2.975

    bestError = None
    bestNum = 0
    params_pso["num_informants"] = 0
    for i in range(40):
        params_pso["num_informants"] = i+1
        error = 0 
        for j in range(10):
          #  print("\nRun ", i)
            experiment2 = Experiment(params_pso, net_layers, path="2in_xor.txt", debugMode=False, sampleMode=True)
            experiment2.run()
            error += experiment2.pso.best.fitness
        avg = error/10
        results.write("\nXor - Avg error for " + str(i) + " informants: " + str(avg))
        if (bestError == None or avg < bestError):
            bestError = avg
            bestNum = i+1
    results.write("\nBest Informants for XOR " + str(bestNum) + " Informants : " + str(bestError)) 

#Complex
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

    params_pso["beta"] = 1.38
    params_pso["gamma"] = 2.72
    

    bestError = None
    bestNum = 0
    params_pso["num_informants"] = 0
    for i in range(40):
        params_pso["num_informants"] = i+1
        error = 0 
        for j in range(10):
          #  print("\nRun ", i)
            experiment2 = Experiment(params_pso, net_complex, path="2in_complex.txt", debugMode=False, sampleMode=True)
            experiment2.run()
            error += experiment2.pso.best.fitness
        avg = error/10
        results.write("\nComplex - Avg error for " + str(i) + " informants: " + str(avg))
        if (bestError == None or avg < bestError):
            bestError = avg
            bestNum = i+1
    results.write("\nBest Informants for Complex " + str(bestNum) + " Informants : " + str(bestError)) 
    results.close()
