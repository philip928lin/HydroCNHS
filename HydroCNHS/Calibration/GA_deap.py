import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from deap import base, creator, tools
import logging
logger = logging.getLogger("HydroCNHS.GA")              

r"""
inputs = {"par_name":    ["a","b","c"],     
          "par_bound":   [[1,2],[1,2],[1,2]],    
          #"par_type":   ["real"]*3,     
          "wd":          r""}    

config = {"min_or_max":         "min",
          "pop_size":           100,    
          "num_ellite":          1,     
          "prob_cross":          0.5,  
          "prob_mut":            0.1,   
          "stochastic":         False,  
          "max_gen":             100,    
          "sampling_method":     "LHC",  
          "drop_record":         False,  
          "paral_cores":         2, 
          "paral_verbose":        0,
          "auto_save":           True,   
          "print_level":         10,      
          "plot":               False    
          }
"""
def scale(individual, bound_scale, lower_bound):
    """individual is 1d ndarray."""
    individual = individual.reshape(bound_scale.shape)
    scaled_individual = np.multiply(individual, bound_scale)
    scaled_individual = np.add(scaled_individual, lower_bound)
    return scaled_individual.flatten()
def descale(individual, bound_scale, lower_bound):
    """individual is 1d array ndarray."""
    individual = individual.reshape(bound_scale.shape)
    descaled_individual = np.subtract(individual, lower_bound)
    descaled_individual = np.divide(descaled_individual, bound_scale)
    return descaled_individual.flatten()
def sample_by_MC(size):
    return np.random.uniform(0, 1, size)
def sample_by_LHC(size):
    # size = [pop_size, num_par]
    pop_size = size[0]
    num_par = size[1]
    pop = np.empty(size)
    for i in range(num_par):
        d = 1.0 / pop_size
        temp = np.empty([pop_size])
        # Uniformly sample in each interval.
        for j in range(pop_size):
            temp[j] = np.random.uniform(low=j * d, high=(j + 1) * d)
        # Shuffle to random order.
        np.random.shuffle(temp)
        # Scale [0,1] to its bound.
        pop[:,i] = temp
    return pop 
def gen_init_pop(creator, size, method="LHC", guess_pop=None):
    # size = [pop_size, num_par]
    pop_size = size[0]
    pop = np.empty(size)
    if guess_pop is not None:
        ass_size = guess_pop.shape[0]
        pop[:ass_size,:] = guess_pop
        # Randomly initialize the rest of population
        size[0] = pop_size - ass_size
    else:
        ass_size = 0
    if method == "MC":
        pop[ass_size:,:] = sample_by_MC(size)
    elif method == "LHC":
        pop[ass_size:,:] = sample_by_LHC(size)
    
    # Convert to DEAP individual objects.
    individuals = []
    for i in range(pop_size):
        individual = pop[i,:]
        individuals.append(creator(individual))
    return individuals
def mut_uniform(individual, prob_mut):
    num_par = len(individual)
    mut = np.random.binomial(n=1, p=prob_mut, size=num_par) == 1
    new_sample = np.random.uniform(0, 1, num_par)
    individual[mut] = new_sample.flatten()[mut] 
    return individual      
def mut_middle(individual, p1, p2, prob_mut):
    num_par = len(individual)
    new_sample = np.random.uniform(0, 1, num_par)
    for i in range(num_par):                           
        rnd = np.random.random()
        if rnd < prob_mut:   
            if p1[i] < p2[i]:
                individual[i] = p1[i] + np.random.random() * (p2[i] - p1[i])  
            elif p1[i] > p2[i]:
                individual[i] = p2[i] + np.random.random() * (p1[i] - p2[i])
            else:
                individual[i] = new_sample[i] 
    return individual

class GA_DEAP(object):
    def __init__(self):
        print("GA Calibration Guide\n"
              +"Step 1: set or load (GA_auto_save.pickle).\nStep 2: run.")
    
    def load(self, GA_auto_save_file):
        with open(GA_auto_save_file, "rb") as f:
            snap_shot = pickle.load(f)
        for key in snap_shot:  # Load back all the previous class attributions.
            setattr(self, key, snap_shot[key])
            
        # Ask for extension of max_gen.  
        config = self.config
        max_gen = config["max_gen"]
        print("Enter the new max_gen (original max_gen = {})".format(max_gen)
              +" or Press Enter to continue.")
        ans1 = input()
        if ans1 != "":
            ans2 = int(ans1)
            if ans2 <= max_gen:
                print("Fail to update MaxGen. Note that new max_gen must be"
                      +" larger than original max_gen. Please reload.")
            else:
                self.config["max_gen"] = ans2

    def set(self, evaluation_func, inputs, config, formatter=None,
            name="Calibration"):
        self.name = name
        self.config = config
        self.inputs = inputs
        self.formatter = formatter
        #self.system_config = loadConfig()
        self.size = (config["pop_size"], len(inputs["par_name"]))
        
        # Continue run setup
        self.done_ini = False                       
        self.current_gen = 0              
        
        # Record
        self.records = {}
        self.solution = {}
        self.summary = {}
        self.summary["max_fitness"] = []
        self.summary["min_fitness"] = []
        self.summary["avg"] = []
        self.summary["std"] = []
        
        # Scale setting 
        bound_scale = []
        lower_bound = []
        par_bound = inputs["par_bound"]
        for i in range(self.size[1]):
        # for i, ty in enumerate(inputs["par_type"]):
            # if ty == "real": Only allow real number for now.
            bound_scale.append(par_bound[i][1] - par_bound[i][0])
            lower_bound.append(par_bound[i][0])
        self.bound_scale = np.array(bound_scale).reshape((-1, self.size[1]))     
        self.lower_bound = np.array(lower_bound).reshape((-1, self.size[1]))        

        # Initialize DEAP setup
        # Setup creator
        if config["min_or_max"] == "min":
            creator.create("Fitness", base.Fitness, weights=(-1.0,))
        else:
            creator.create("Fitness", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray,
                            fitness=creator.Fitness)
        # Setup toolbox
        tb = base.Toolbox()
        tb.register("population", gen_init_pop, creator.Individual)
        tb.register("evaluate", evaluation_func)
        tb.register("scale", scale, bound_scale=self.bound_scale,
                         lower_bound=self.lower_bound)
        tb.register("descale", descale, bound_scale=self.bound_scale,
                         lower_bound=self.lower_bound)
        tb.register("crossover", tools.cxUniform)
        tb.register("mutate_uniform", mut_uniform)
        tb.register("mutate_middle", mut_middle)
        tb.register("select", tools.selRoulette)
        tb.register("ellite", tools.selBest)
        self.tb = tb
        # Create calibration folder under WD
        self.cali_wd = os.path.join(inputs["wd"], name)
        # Create cali_wd directory
        if os.path.isdir(self.cali_wd) is not True:
            os.mkdir(self.cali_wd)
        else:
            logger.warning("Current calibration folder exists."
                           +" Default to overwrite the folder!"
                           +"\n{}".format(self.cali_wd))
        #---------------------------------------

    def run(self, guess_pop=None):
        
        # Start timer
        self.start_time = time.monotonic()
        self.elapsed_time = 0
        
        config = self.config
        paral_cores = config["paral_cores"]
        paral_verbose = config["paral_verbose"]
        formatter = self.formatter
        cali_wd = self.cali_wd
        max_gen = config["max_gen"]
        size = self.size
        stochastic = config["stochastic"]
        tb = self.tb
        
        # Initialization
        if self.done_ini is False:
            pop = tb.population(self.size, config["sampling_method"],
                                guess_pop)
            self.done_ini = True
            scaled_pop = list(map(tb.scale, pop))
            # Note np.array(scaled_pop[k]) is necessary for serialization.
            # Use joblib instead of DEAP document of Scoop or muliprocessing, 
            # so we don't have to run in external terminal.
            fitnesses = Parallel(n_jobs=paral_cores, verbose=paral_verbose) \
                                ( delayed(tb.evaluate)\
                                    (np.array(scaled_pop[k]), formatter,
                                     (cali_wd, self.current_gen, k)) \
                                    for k in range(len(scaled_pop)) )        
                                    
            # Note that we assign fitness to original pop not the scaled_pop.
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            self.find_best_and_record(pop)
        else:
            pop = self.records[self.current_gen-1]
        
        # Iteration
        prob_cross = config["prob_cross"]
        prob_mut = config["prob_mut"]
        while self.current_gen <= max_gen:
            
            # Select the next generation individuals
            parents = tb.select(pop, size[0])
            # Clone the selected individuals
            offspring = list(map(tb.clone, parents))
            
            # Apply crossover and mutation on the offspring
            for p1, p2, child1, child2 in zip(parents[::2], parents[1::2],
                                              offspring[::2], offspring[1::2]):
                if np.random.uniform() < prob_cross:
                    tb.crossover(child1, child2, prob_cross)
                if np.random.uniform() < prob_mut:
                    tb.mutate_uniform(child1, prob_mut)
                    tb.mutate_middle(child2, p1, p2, prob_mut)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Replace with ellite
            for i, e in enumerate(self.ellites):
                offspring[i] = e
            
            # Evaluate the individuals with an invalid fitness
            if stochastic:
                invalid_ind = offspring
            else:
                invalid_ind = [ind for ind in offspring \
                               if not ind.fitness.valid]
            
            scaled_pop = list(map(tb.scale, invalid_ind))
            fitnesses = Parallel(n_jobs=paral_cores, verbose=paral_verbose) \
                                ( delayed(tb.evaluate)\
                                    (np.array(scaled_pop[k]), formatter,
                                     (cali_wd, self.current_gen, k)) \
                                    for k in range(len(invalid_ind)) )
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring
            self.find_best_and_record(pop)
        print("\nGA done!\n")
        
    def find_best_and_record(self, pop):
        # Select ellite and save.
        tb = self.tb
        config = self.config
        num_ellite = config["num_ellite"]
        ellites = tb.ellite(pop, num_ellite)
        
        self.ellites = list(map(tb.clone, ellites))
        self.solution = tb.scale(tb.clone(ellites[0]))
        self.records[self.current_gen] = pop
        if config["drop_record"]:
            # Delete previous generation's record 
            self.records.pop(self.current_gen-1,"")
        
        self.current_gen += 1
        
        if config["auto_save"]:
            self.auto_save()
            
        # Calculate indicators
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        elapsed_time = time.monotonic() - self.start_time
        self.summary["elapsed_time"] = time.strftime(
            "%H:%M:%S", time.gmtime(elapsed_time))
        self.summary["max_fitness"].append(max(fits))
        self.summary["min_fitness"].append(min(fits))
        self.summary["avg"].append(mean)
        self.summary["std"].append(std)
        
        if ((self.current_gen-1) % config["print_level"] == 0 
            or self.current_gen > config["max_gen"]):
            print("\n=====Generation {}=====".format(self.current_gen-1))
            print("  Elapsed time %s" % elapsed_time)
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
            
            if config["plot"]:
                # Plot progress
                fig, ax1 = plt.subplots()
                ax2 = ax1.twinx()
                if config["min_or_max"] == "max":
                    fitness = self.summary["max_fitness"]
                    ax1.set_ylabel("Fitness (Max)")
                else:
                    fitness = self.summary["min_fitness"]
                    ax1.set_ylabel("Fitness (Min)")
                x = np.arange(len(fitness))
                lns1 = ax1.plot(x, fitness, label="Fitness", linewidth=2,
                                color="black", marker=".")        
                lns2 = ax2.plot(x, self.summary["std"], label="Fitness std",
                                linewidth=2, color="grey", linestyle="--",
                                marker="x")  
                ax2.set_ylabel("Fitness standard deviation")
        
                ax1.set_title(
                    self.name + "  [{}]".format(self.summary["elapsed_time"]))
                ax1.set_xlim([0, config["max_gen"]])
                ax1.set_xlabel("Generation")
                
                lns = lns1+lns2
                labs = [l.get_label() for l in lns]
                ax1.legend(lns, labs)
                plt.tight_layout()
                filename = os.path.join(self.cali_wd,
                                        "Fitness_" + self.name + ".png")
                fig.savefig(filename, dpi=300)
                if config["paral_cores"] == 1:
                    plt.show()
                plt.close()
        
    def auto_save(self):
        cali_wd = self.cali_wd
        snap_shot = self.__dict__
        with open(os.path.join(cali_wd, "GA_auto_save.pickle"),
                  'wb') as outfile:
            pickle.dump(snap_shot, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        return None