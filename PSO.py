import numpy as np
import random as rand

class Particle:
    def __init__(self, position, velocity, ideal):
        #the index of a position or velocity in the list corresponds to a timestamp
        self.position_list = [position]
        self.velocity_list = [velocity]
        self.position = position
        self.velocity = velocity
        self.ideal = ideal
        self.best = position
        self.fitness = np.linalg.norm(self.ideal-self.position) 
        self.best_fitness = self.fitness

    def update_position(self, new_pos):
        self.position_list.append(new_pos)
        self.position = new_pos
        
    def update_velocity(self, new_vel):
        self.velocity_list.append(new_vel)
        self.velocity = new_vel

    def asses_fitness(self):
        self.fitness = np.linalg.norm(self.ideal-self.position) #numpy implementation of euclidean distance   
        #otherwise get distance from ideal and see if it is better than current best
        if (self.fitness < self.best_fitness):
            self.best = self.position
            self.best_fitness = self.fitness
                
    def informants_best(self):
        inf_best = self.informants[0]        
        for inf in self.informants:
            if (inf.best_fitness < inf_best.best_fitness):
                inf_best = inf

        return inf_best.best


class PSO:
    def __init__ (self, swarmsize, alpha, beta, gamma, delta, jumpsize, ideal, num_informants, max_runs, boundary) :
        self.swarmsize = swarmsize #size of the swarm
        self.alpha = alpha #proportion of velocity to be retained
        self.beta = beta #proportion of personal best to be retained
        self.gamma = gamma #proportion of the 'informants' best to be retained
        self.delta = delta #proportion of global best to be retained
        self.jumpsize = jumpsize #jumpsize
        self.num_informants = num_informants #number of randomly selected informants per particle
        self.ideal = ideal
        self.max_runs = max_runs
        self.best = None #this is probably not a good idea
        self.boundary = boundary
        self.best_count = 0

    def generate_particles(self):
        self.particles = []
        for i in range(self.swarmsize):
            particle_pos = np.array([])
            particle_vel = np.array([])
            
            for j in range(len(self.ideal)):
                particle_pos = np.append(particle_pos, np.random.uniform(-10.0, 10.0))
                particle_vel = np.append(particle_vel, np.random.uniform(-10.0, 10.0))

            new_particle = Particle(particle_pos, particle_vel, self.ideal)
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
                self.best = Particle(particle.position, particle.velocity, particle.ideal)
            elif particle.fitness < self.best.fitness:
                self.best.update_position(particle.position)
                self.best.update_velocity(particle.velocity)
                self.best.asses_fitness()
                
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
                b_val = b*(part_best[i] - particle_vel[i]) 
                c_val = c*(inf_best[i] -particle_vel[i])
                d_val = d*(best_pos[i] -particle_vel[i])
                
                new_vel.append(self.alpha*particle_vel[i] + b_val + c_val + d_val)                
                        
            particle.update_velocity(new_vel)
            #print(new_vel)
            
    def update_positions(self):
        for particle in self.particles:
            new_pos = particle.position + (np.dot(self.jumpsize, particle.velocity)) #use dot to multiply jumpsize by velo array
            
            #if it exceeds the boundary reverse velocity
            for i in range(len(new_pos)):
                if (new_pos[i] > self.boundary[i] or new_pos[i] < -self.boundary[i]):
                    particle.velocity[i] = -particle.velocity[i]
                    
            new_pos = particle.position + (np.dot(self.jumpsize, particle.velocity))
                    
            
            particle.update_position(new_pos)
         #   print(particle.position)
            
    def run_algo(self):
        self.generate_particles()
        self.assign_informants()
        runs = 0
        while (self.best == None or (self.best.fitness > 0.001 and runs < self.max_runs)):
            self.asses_fitness()
            self.update_velocity()
            self.update_positions()
            runs += 1
        print(runs)
            
            
swarmsize = 1000
alpha = 1
beta = 1
gamma = 1
delta = 1
jumpsize = 0.5
ideal = [0.0,0.0]
boundary = [5,5]
num_informants = 10
max_runs = 1000

my_pso = PSO(swarmsize, alpha, beta, gamma, delta, jumpsize, ideal, num_informants, max_runs, boundary)
my_pso.run_algo()

#my_pso.generate_particles()




            






            