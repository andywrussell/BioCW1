import numpy as np

class Particle:
    def __init__(self, position, velocity):
        #the index of a position or velocity in the list corresponds to a timestamp
        self.position_list = [position]
        self.velocity_list = [velocity]
        self.position = position
        self.velocity = velocity
    
    def update_particle(self, new_pos, new_vel):
        self.update_position(new_pos)
        self.update_velocity(new_vel)

    def update_position(self, new_pos):
        self.position_list.append(new_pos)
        self.position = new_pos

    def update_velocity(self, new_vel):
        self.velocity_list.append(new_vel)
        self.velocity = new_vel


class PSO:
    def __init__ (self, swarmsize, alpha, beta, gamma, sigma, jumpsize) :
        self.swarmsize = swarmsize #size of the swarm
        self.alpha = alpha #proportion of velocity to be retained
        self.beta = beta #proportion of personal best to be retained
        self.gamma = gamma #proportion of the 'informants' best to be retained
        self.sigma = sigma #proportion of global best to be retained
        self.jumpsize = jumpsize #jumpsize

    def generate_particles(self):
        counter = 0
        self.particles = []
        while(counter < self.swarmsize):
            particle_pos = np.random.random_sample()
            particle_vel = np.random.random_sample()
            new_particle = Particle(particle_pos, particle_vel)
            self.particles.append(new_particle)
            counter += 1

    def run_algo(self):
        self.best = null


            