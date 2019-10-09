import math

def NullActivator(in_val) : 
    return 0

def SigmoidActivator(in_val) :
    return 1 / (1 - math.exp(-in_val))

def HyperbolicActivator(in_val) : 
    return math.tanh(in_val)

def CosineActivator(in_val) :
    return math.cos(in_val)

##This one doesnt work
def GaussianActivator(in_val) : 
    return math.exp(-((in_val**2)/2))


activation_dict = {
    0: NullActivator,
    1: SigmoidActivator,
    2: HyperbolicActivator,
    3: CosineActivator,
    4: GaussianActivator
}