# Optimizing a Neural Network Parameters with PSO

## Installation

To run the program simply install the requirements in `requirements.txt` and run `main.py`, which will run some default experiments.

## Documentation

### **Create a network**

Create a neural net with hidden layers.

```
import ANN.NeuralNet

# Create a neural network with 2 layers with 4 nodes and 1 output.

network = NeuralNet()
network.add_layer(input_count=1, node_count=4, activations=[0,2,3,4])
network.add_layer(input_count=4, node_count=1, activations=[2])
```

Create a NeuralNet with `NetworkGenerator`. This class lets you specify the network structure you want and then create as many instances of that network as you want.
```
from ANN.networkGenerator import NetworkGenerator

# Start the generator and add desired layer size
net_generator = NetworkGenerator()
net_generator.add_layer(input_count=1, node_count=4, activations=[0,2,3,4])
net_generator.add_layer(input_count=4, node_count=1, activations=[2])

# Generate a neural net with those layer parameters
network = net_generator.generate_network()
```
### **Run the PSO**
Create a PSO algorithm to improve your network parameters.

```
from PSO.pso import PSO

# Create a network generator with 2 hidden layers.
net_generator = NetworkGenerator()
net_generator.add_layer(input_count=1, node_count=4, activations=[0,2,3,4])
net_generator.add_layer(input_count=4, node_count=1, activations=[2])


# Pass the generator to the PSO class along with other parameters.
params_pso = PSO(
        net_generator = net_generator,
        swarmsize = 40,
        alpha = 1,
        beta = 1.18,
        gamma = 2.92,
        delta = 0,
        jumpsize = 1,
        act_bound = 5,
        weight_bound = 10,
        bound_strat = 1,
        num_informants = 3,
        vel_range = 1,
        max_runs = 1000,
        informants_strat = 0
    )

# Run the pso algorithm
pso.run_algo()
```

## **Create an Experiment**

Create an experiment for different parameters of the PSO and the Neural Net.

```
from experiments import Experiment


# Set the paramaters for the pso
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

# Set the parameters for Neural Net
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

# Create the experiment and run it.
experiment = Experiment(params_pso, net_layers, path="1in_linear.txt")
experiment.run()
```