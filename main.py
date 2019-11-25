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
from Tests import baseline, swarmsize, alpha, beta, informants, final
from Tests.networks.net_architectures import net_simple_1, net_simple_2, net_complex
import os

#beta.run_beta()

#informants.run_informant_count()
#final.run_final()
