import math

def NullActivator(in_val) : 
    return 0

def SigmoidActivator(in_val) :
    return 1 / (1 + math.exp(-abs(in_val)))

def HyperbolicActivator(in_val) : 
    return math.tanh(in_val)

def CosineActivator(in_val) :
    return math.cos(in_val)


def GaussianActivator(in_val) : 
    return math.exp(-((in_val**2)/2))


def ReLUActivator(in_val) :
    if in_val < 0:
        return 0

    return in_val


activation_dict = {
    0: NullActivator,
    1: SigmoidActivator,
    2: HyperbolicActivator,
    3: CosineActivator,
    4: GaussianActivator,
    5: ReLUActivator
}

activation_index = {
    0: "Null Activator",
    1: "Sigmoid Activator",
    2: "Hyperbolic Activator",
    3: "Cosine Activator",
    4: "Gaussian Activator",
    5: "ReLU Activator"
}