from PSO.particle import Particle
import numpy as np
from tqdm import tqdm

class PSO:
    def __init__ (self, net_generator, swarmsize, alpha, beta, gamma, delta, jumpsize, ideal, inputs, num_informants, max_runs, act_bound, weight_bound = 0, bound_strat = 3) :
        self.net_generator = net_generator # Class that returns a neural net with given layers.
        self.swarmsize = swarmsize #size of the swarm
        self.alpha = alpha #proportion of velocity to be retained
        self.beta = beta #proportion of personal best to be retained
        self.gamma = gamma #proportion of the 'informants' best to be retained
        self.delta = delta #proportion of global best to be retained
        self.jumpsize = jumpsize #jumpsize
        self.num_informants = num_informants #number of randomly selected informants per particle
        self.ideal = ideal
        self.inputs = inputs
        self.max_runs = max_runs
        self.best = None #this is probably not a good idea
        self.act_bound = act_bound #activation boundary
        self.weight_bound = weight_bound #boundary for weigths
        self.bound_strat = bound_strat #0 for no boundary, 1 for ignore move, 2 for set to boundary, 3 for reflect
        self.best_count = 0

    def generate_particles(self):
        self.particles = []
        for i in range(self.swarmsize):
            
            # my_test_input = np.array([1])
            """
            network = NeuralNet(error_function=MSE)
            network.add_layer(input_count=2 , node_count=4, activations=[0,1,2,2])
            network.add_layer(input_count=4 , node_count=1, activations=[0])
            """
            network = self.net_generator.generate_network()
            network.flatten_net()

            particle_pos = network.net_as_vector 
            network.unflatten_net()
            particle_vel = np.array([])

            for j in range(len(particle_pos)):
                particle_vel = np.append(particle_vel, np.random.uniform(-1, 1))
                
            new_particle = Particle(network, particle_pos, particle_vel, self.ideal, self.inputs)
            self.particles.append(new_particle)


    def assign_informants(self):
        for particle in self.particles:
            particle.informants = []
            for i in range(self.num_informants):
                rand_inf = np.random.choice(self.particles)
                particle.informants.append(rand_inf)
                
    def asses_fitness(self):
        for particle in self.particles:
            particle.asses_fitness()
            if self.best == None:
                self.best = Particle(particle.network, particle.position, particle.velocity, particle.ideal, particle.inputs)
            elif particle.fitness < self.best.fitness:
                self.best.network = particle.network
                self.best.update_position(particle.position)
                self.best.update_velocity(particle.velocity)
                self.best.fitness = particle.fitness
                self.best.best_fitness = particle.best_fitness
                self.best.best = particle.position
                self.best.outputs = particle.outputs
                
    def update_velocity(self): 
        best_pos = self.best.position                        
        
        for particle in self.particles:
            part_best = particle.best
            inf_best = particle.informants_best()
            particle_vel = particle.velocity
            new_vel = []

            b = np.random.uniform(0, self.beta)
            c = np.random.uniform(0, self.gamma)
            d = np.random.uniform(0, self.delta)
                        
            for i in range(len(particle_vel)):                
                b_val = b*(part_best[i] - particle.position[i]) 
                c_val = c*(inf_best[i] -particle.position[i])
                d_val = d*(best_pos[i] -particle.position[i])
                
                new_vel.append(self.alpha*particle_vel[i] + b_val + c_val + d_val)                
                        
            particle.update_velocity(new_vel)
            
    def update_positions(self):

        for particle in self.particles:
            new_pos = particle.position + (np.dot(self.jumpsize, particle.velocity)) #use dot to multiply jumpsize by velo array
            
            act_idx = particle.network.activation_idx
            for i in range(len(new_pos)):
                #if we are at an activation function impose boundary
                boundary = self.weight_bound
                is_activation = False
                if i in act_idx:
                    boundary = self.act_bound
                    is_activation = True
                    
                if boundary > 0:
                    new_pos[i], particle.velocity[i] = self.check_boundary(new_pos[i], particle.position[i], particle.velocity[i], boundary, is_activation)

                    # if new_pos[i] > boundary:
                    #     diff = new_pos[i] - boundary
                    #     particle.velocity[i] = -particle.velocity[i]
                    #     new_pos[i] = boundary - diff
                    # elif new_pos[i] < -boundary:
                    #     diff = abs(new_pos[i]) - boundary
                    #     new_pos[i] = (-boundary) + diff
                    #     particle.velocity[i] = -particle.velocity[i]
                    #     new_pos[i] = -boundary

                         
            particle.update_position(new_pos)
         #   print(particle.position)

    def check_boundary(self, new_pos, old_pos, vel, boundary, is_activation):
        if self.bound_strat == 0: #ignore boundary
            return new_pos, vel

        elif self.bound_strat == 1: #ignore move
            return old_pos, 0

        elif self.bound_strat == 2: #set to boundary
            if new_pos > boundary:
                new_pos = boundary
            elif new_pos < -boundary:
                new_pos = -boundary
            return new_pos, -vel

        elif self.bound_strat == 3: #bounce off boundary
            if new_pos > boundary:
                diff = new_pos - boundary
                new_pos = boundary - diff
            elif new_pos < -boundary:
                diff = abs(new_pos) - boundary
                new_pos = (-boundary) + diff
            return new_pos, -vel


    def run_algo(self):
        self.generate_particles()
        self.assign_informants()
        run = 1

        progress_bar = tqdm(range(self.max_runs))
        for i in progress_bar:
            if self.best == None or (self.best.fitness > 0.001):
                self.asses_fitness()
                self.update_velocity()
                self.update_positions()
                cur_best = self.best.fitness
                progress_bar.set_description(" Run {}/{} | Best fitness = {}".format(run, self.max_runs, round(self.best.fitness, 4)))
                run += 1
            else:
                break;            
            

            
            
