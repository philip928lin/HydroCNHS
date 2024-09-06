import os
import time
import pickle
import numpy as np
from operator import attrgetter
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from deap import base, creator, tools
import logging

logger = logging.getLogger("HydroCNHS.GA")

r"""
inputs = {"par_name":    ["a","b","c"],
          "par_bound":   [[0,1],[0,1],[0,1]],
          "wd":          r""}

config = {"min_or_max":         "min",
          "pop_size":           100,
          "num_ellite":         1,
          "prob_cross":         0.5,
          "prob_mut":           0.1,
          "stochastic":         False,
          "max_gen":            100,
          "sampling_method":    "LHC",
          "drop_record":        False,
          "paral_cores":        -1,
          "paral_verbose":      1,
          "auto_save":          True,
          "print_level":        10,
          "plot":               True
          }
"""


def scale(individual, bound_scale, lower_bound):
    """individual is 1d ndarray."""
    individual = individual.reshape(bound_scale.shape)
    scaled_individual = np.multiply(individual, bound_scale)
    scaled_individual = np.add(scaled_individual, lower_bound)
    return scaled_individual.flatten()


def descale(individual, bound_scale, lower_bound):
    """individual is 1d ndarray."""
    individual = individual.reshape(bound_scale.shape)
    descaled_individual = np.subtract(individual, lower_bound)
    descaled_individual = np.divide(descaled_individual, bound_scale)
    return descaled_individual.flatten()


def sample_by_MC(size, rn_gen_gen):
    return rn_gen_gen.uniform(0, 1, size)


def sample_by_LHC(size, rn_gen_gen):
    # size = [pop_size, num_par]
    pop_size = size[0]
    num_par = size[1]
    pop = np.empty(size)
    for i in range(num_par):
        d = 1.0 / pop_size
        temp = np.empty([pop_size])
        # Uniformly sample in each interval.
        for j in range(pop_size):
            temp[j] = rn_gen_gen.uniform(low=j * d, high=(j + 1) * d)
        # Shuffle to random order.
        rn_gen_gen.shuffle(temp)
        # Scale [0,1] to its bound.
        pop[:, i] = temp
    return pop


def gen_init_pop(creator, size, method="LHC", guess_pop=None, rn_gen_gen=None):
    # size = [pop_size, num_par]
    pop_size = size[0]
    pop = np.empty(size)
    if guess_pop is not None:
        ass_size = guess_pop.shape[0]
        pop[:ass_size, :] = guess_pop
        # Randomly initialize the rest of population
        size[0] = pop_size - ass_size
    else:
        ass_size = 0
    if method == "MC":
        pop[ass_size:, :] = sample_by_MC(size, rn_gen_gen)
    elif method == "LHC":
        pop[ass_size:, :] = sample_by_LHC(size, rn_gen_gen)

    # Convert to DEAP individual objects.
    individuals = []
    for i in range(pop_size):
        individual = pop[i, :]
        individuals.append(creator(individual))
    return individuals


def mut_uniform(individual, prob_mut, rn_gen_gen):
    num_par = len(individual)
    mut = rn_gen_gen.binomial(n=1, p=prob_mut, size=num_par) == 1
    new_sample = rn_gen_gen.uniform(0, 1, num_par)
    individual[mut] = new_sample.flatten()[mut]
    return individual


def mut_middle(individual, p1, p2, prob_mut, rn_gen_gen):
    num_par = len(individual)
    new_sample = rn_gen_gen.uniform(0, 1, num_par)
    for i in range(num_par):
        rnd = rn_gen_gen.random()
        if rnd < prob_mut:
            if p1[i] < p2[i]:
                individual[i] = p1[i] + rn_gen_gen.random() * (p2[i] - p1[i])
            elif p1[i] > p2[i]:
                individual[i] = p2[i] + rn_gen_gen.random() * (p1[i] - p2[i])
            else:
                individual[i] = new_sample[i]
    return individual


def selRoulette(individuals, k, rn_gen_gen, fit_attr="fitness"):
    # From DEAP
    s_inds = sorted(individuals, key=attrgetter(fit_attr), reverse=True)
    sum_fits = sum(getattr(ind, fit_attr).values[0] for ind in individuals)
    chosen = []
    for i in range(k):
        u = rn_gen_gen.random() * sum_fits
        sum_ = 0
        for ind in s_inds:
            sum_ += getattr(ind, fit_attr).values[0]
            if sum_ > u:
                chosen.append(ind)
                break
    return chosen


def cxUniform(ind1, ind2, indpb, rn_gen_gen):
    # From DEAP
    size = min(len(ind1), len(ind2))
    for i in range(size):
        if rn_gen_gen.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2


##### Set DEAP
# Build creator
# This will create new class for the deap.creator library, which cannot be
# pickled.
creator.create("Fitness_min", base.Fitness, weights=(-1.0,))
creator.create("Fitness_max", base.Fitness, weights=(1.0,))
creator.create("Individual_min", np.ndarray, fitness=creator.Fitness_min)
creator.create("Individual_max", np.ndarray, fitness=creator.Fitness_max)

tb = base.Toolbox()

tb.register("crossover", cxUniform)  # apply customized rn_gen
tb.register("mutate_uniform", mut_uniform)
tb.register("mutate_middle", mut_middle)
tb.register("select", selRoulette)  # apply customized rn_gen
tb.register("ellite", tools.selBest)


class GA_DEAP(object):
    def __init__(self, evaluation_func, rn_gen=None):
        """Initialize the GA calibration object.

        Note that this GA algorithm only allows to calibrate real numbers.

        Parameters
        ----------
        evaluation_func : function
            Evaluation function. EX:
            def evaluation(individual, info):
                return (fitness,)
            where info = (cali_wd, current_generation, ith_individual,
            formatter, rn_gen)
        rn_gen : object, optional
            Random number generator created by create_rn_gen(), by default None.
            If given, randomness of the designed model is controled by rn_gen.
            We encourage user to assign it to maintain the reproducibility of
            the stochastic simulation.
        """
        print(
            "GA Calibration Guide\n"
            + "Step 1: set or load (GA_auto_save.pickle).\nStep 2: run."
        )
        if rn_gen is None:
            # Assign a random seed.
            seed = np.random.randint(0, 100000)
            self.rn_gen = np.random.default_rng(seed)
        else:
            # User-provided rn generator
            self.rn_gen = rn_gen
            self.ss = rn_gen.bit_generator._seed_seq
            logger.info("User-provided random number generator is assigned.")
        tb.register("evaluate", evaluation_func)

    def load(self, GA_auto_save_file, max_gen=None):
        """Load save pickle file (i.e., continue previous run).

        Parameters
        ----------
        GA_auto_save_file : str
            Filename.
        max_gen : int, optional
            This allow user to increase max_gen for continuing calibration for
            a longer searching, by default None.
        """
        with open(GA_auto_save_file, "rb") as f:
            snap_shot = pickle.load(f)
        for key in snap_shot:  # Load back all the previous class attributions.
            setattr(self, key, snap_shot[key])

        # Ask for extension of max_gen.
        config = self.config
        max_gen_org = config["max_gen"]
        if max_gen is None:
            print(
                "Enter the new max_gen (original max_gen = {})".format(max_gen)
                + " or Press Enter to continue."
            )
            ans1 = input()
        else:
            ans1 = max_gen
        if ans1 != "":
            ans2 = int(ans1)
            if ans2 <= max_gen_org:
                print(
                    "Fail to update MaxGen. Note that new max_gen must be"
                    + " larger than original max_gen. Please reload."
                )
            else:
                self.config["max_gen"] = ans2
                # Add random seed if increased max gen.
                self.rng_seeds += self.ss.spawn(ans2 - max_gen_org)
        # Add toolbox
        if config["min_or_max"] == "min":
            tb.register("population", gen_init_pop, creator.Individual_min)
        else:
            tb.register("population", gen_init_pop, creator.Individual_max)
        tb.register(
            "scale", scale, bound_scale=self.bound_scale, lower_bound=self.lower_bound
        )
        tb.register(
            "descale",
            descale,
            bound_scale=self.bound_scale,
            lower_bound=self.lower_bound,
        )

    def set(self, inputs, config, formatter=None, name="Calibration"):
        """Setup the GA calibration.

        Parameters
        ----------
        inputs : dict
            Calibration input dictionary generated by Convertor. Or, get the
            template by calling get_inputs_template().
        config : dict
            Calibration configuration dictionary. Get the template by calling
            get_config_template().
        formatter : dict, optional
            Formatter generated by Convertor, by default None.
        name : str, optional
            Name of the calibration, by default "Calibration".
        """
        self.name = name
        self.config = config
        self.inputs = inputs
        self.formatter = formatter
        # self.system_config = loadConfig()
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

        # Random number generators' seed for each generation (from 0 to max_gen)
        self.rng_seeds = self.ss.spawn(config["max_gen"] + 1)

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
        # Setup creator (I think this will not be saved in pickle)
        # https://stackoverflow.com/questions/66894090/storing-deap-models-as-pickle-objects
        # My solution is to build creator globally on top of the script.

        # Add toolbox
        if config["min_or_max"] == "min":
            tb.register("population", gen_init_pop, creator.Individual_min)
        else:
            tb.register("population", gen_init_pop, creator.Individual_max)
        tb.register(
            "scale", scale, bound_scale=self.bound_scale, lower_bound=self.lower_bound
        )
        tb.register(
            "descale",
            descale,
            bound_scale=self.bound_scale,
            lower_bound=self.lower_bound,
        )

        # Create calibration folder under WD
        self.cali_wd = os.path.join(inputs["wd"], name)
        # Create cali_wd directory
        if os.path.isdir(self.cali_wd) is not True:
            os.mkdir(self.cali_wd)
        else:
            logger.warning(
                "Current calibration folder exists."
                + " Default to overwrite the folder!"
                + "\n{}".format(self.cali_wd)
            )
        # ---------------------------------------

    def run_individual(self, individual="best", name="best"):
        """Run the evaluation for a given individual.

        Warning! run_individual() does not generantee the same rn_gen will be
        assign to the evaluation, but the same one will be used for
        run_individual()

        Parameters
        ----------
        individual : 1darray, optional
            Individual or solution, by default "best".
        name : str, optional
            This will be sent to the evaluation function through info =
            (cali_wd, name, name, formatter, rn_gen), by default "best".
        """

        if individual == "best":
            sol = self.solution
        else:
            sol = np.array(individual)
        formatter = self.formatter
        cali_wd = self.cali_wd
        # Warning! Does not generantee the same rn_gen will be assign to the
        # evaluation, but the same one will be used for run_individual()
        rn_gen_pop = self.gen_rn_gens(
            self.rng_seeds[self.current_gen - 1], self.size[0]
        )
        fitness = tb.evaluate(sol, (cali_wd, name, name, formatter, rn_gen_pop[0]))

        print("Fitness: {}".format(fitness))

    def gen_rn_gens(self, seed, size):
        # Create rn_gen for each individual in the pop with predefined
        # generation specific seed.
        rn_gen_gen = np.random.default_rng(seed)
        ind_seeds = rn_gen_gen.bit_generator._seed_seq.spawn(size + 1)
        rn_gen_pop = [np.random.default_rng(s) for s in ind_seeds]
        # rn_gen for selection, crossover, mutation for each generation
        self.rn_gen_gen = rn_gen_gen
        return rn_gen_pop

    def run(self, guess_pop=None):
        """Run calibration.

        Parameters
        ----------
        guess_pop : 2darray, optional
            Assigned initial guesses, by default None. guess_pop has the size =
            [number of guesses, number of parameters]
        """
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

        # Initialization
        if self.done_ini is False:
            # self.rng_seeds[0] will be used for initialize population as well
            # as form pop for the first generation.
            rn_gen_pop = self.gen_rn_gens(self.rng_seeds[self.current_gen], size[0])

            pop = tb.population(
                self.size, config["sampling_method"], guess_pop, self.rn_gen_gen
            )
            self.done_ini = True
            scaled_pop = list(map(tb.scale, pop))
            # Note np.array(scaled_pop[k]) is necessary for serialization.
            # Use joblib instead of DEAP document of Scoop or muliprocessing,
            # so we don't have to run in external terminal.

            fitnesses = Parallel(n_jobs=paral_cores, verbose=paral_verbose)(
                delayed(tb.evaluate)(
                    np.array(scaled_pop[k]),
                    (cali_wd, self.current_gen, k, formatter, rn_gen_pop[k]),
                )
                for k in range(len(scaled_pop))
            )

            # Note that we assign fitness to original pop not the scaled_pop.
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            self.find_best_and_record(pop)
        else:  # Load previous run
            pop = self.records[self.current_gen - 1]
            # To continue the random sequence from previous run
            # (do ga things using seed from last generation)
            self.rn_gen_gen = np.random.default_rng(
                self.rng_seeds[self.current_gen - 1]
            )

        # Iteration
        prob_cross = config["prob_cross"]
        prob_mut = config["prob_mut"]
        while self.current_gen <= max_gen:
            # Select the next generation individuals
            parents = tb.select(pop, size[0], self.rn_gen_gen)
            # Clone the selected individuals
            offspring = list(map(tb.clone, parents))

            # Apply crossover and mutation on the offspring
            for p1, p2, child1, child2 in zip(
                parents[::2], parents[1::2], offspring[::2], offspring[1::2]
            ):
                # Keep the parent with some probability prob_cross
                # if np.random.uniform() < prob_cross:
                #     tb.crossover(child1, child2, prob_cross)

                # Crossover must happen
                tb.crossover(child1, child2, prob_cross, self.rn_gen_gen)
                if self.rn_gen_gen.uniform() < prob_mut:
                    tb.mutate_uniform(child1, prob_mut, self.rn_gen_gen)
                    tb.mutate_middle(child2, p1, p2, prob_mut, self.rn_gen_gen)

                # Delete fitnesses
                del child1.fitness.values
                del child2.fitness.values

            # Replace with ellite (with its fitness, no rerun)
            for i, e in enumerate(self.ellites):
                offspring[i] = e

            # Evaluate the individuals with an invalid fitness
            if stochastic:
                invalid_ind = offspring
            else:
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            scaled_pop = list(map(tb.scale, invalid_ind))
            rn_gen_pop = self.gen_rn_gens(self.rng_seeds[self.current_gen], size[0])
            fitnesses = Parallel(n_jobs=paral_cores, verbose=paral_verbose)(
                delayed(tb.evaluate)(
                    np.array(scaled_pop[k]),
                    (cali_wd, self.current_gen, k, formatter, rn_gen_pop[k]),
                )
                for k in range(len(invalid_ind))
            )
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring
            self.find_best_and_record(pop)
        print("\nGA done!\n")

    def find_best_and_record(self, pop):
        # Select ellite and save.
        config = self.config
        num_ellite = config["num_ellite"]
        ellites = tb.ellite(pop, num_ellite)

        self.ellites = list(map(tb.clone, ellites))
        self.solution = np.array(tb.scale(tb.clone(ellites[0])))
        self.records[self.current_gen] = list(map(tb.clone, pop))
        if config["drop_record"]:
            # Delete previous generation's record
            self.records.pop(self.current_gen - 1, "")

        self.current_gen += 1

        # Calculate indicators
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean**2) ** 0.5

        elapsed_time = time.monotonic() - self.start_time
        self.summary["elapsed_time"] = time.strftime(
            "%H:%M:%S", time.gmtime(elapsed_time)
        )
        self.summary["max_fitness"].append(max(fits))
        self.summary["min_fitness"].append(min(fits))
        self.summary["avg"].append(mean)
        self.summary["std"].append(std)

        # Auto save
        if config["auto_save"]:
            self.auto_save()

        if (self.current_gen - 1) % config[
            "print_level"
        ] == 0 or self.current_gen > config["max_gen"]:
            print("\n=====Generation {}=====".format(self.current_gen - 1))
            print("  Elapsed time %s" % self.summary["elapsed_time"])
            print("  Min %s" % round(min(fits), 5))
            print("  Max %s" % round(max(fits), 5))
            print("  Avg %s" % round(mean, 5))
            print("  Std %s" % round(std, 5))

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
                lns1 = ax1.plot(
                    x,
                    fitness,
                    label="Fitness",
                    linewidth=2,
                    color="black",
                    marker=".",
                    zorder=2,
                )
                lns2 = ax2.plot(
                    x,
                    self.summary["std"],
                    label="Fitness std",
                    linewidth=2,
                    color="grey",
                    linestyle="--",
                    marker="x",
                    zorder=1,
                )
                ax2.set_ylabel("Fitness standard deviation")

                ax1.set_title(self.name + "  [{}]".format(self.summary["elapsed_time"]))
                ax1.set_xlim([0, config["max_gen"]])
                ax1.set_xlabel("Generation")

                lns = lns1 + lns2
                labs = [l.get_label() for l in lns]
                ax1.legend(lns, labs)
                plt.tight_layout()
                filename = os.path.join(self.cali_wd, "Fitness_" + self.name + ".png")
                try:
                    fig.savefig(filename, dpi=300)
                except Exception as e:
                    logger.error(e)
                    print("File might be in-use. => Show in console.")
                    plt.show()
                if config["paral_cores"] == 1:
                    plt.show()
                plt.close()

    def auto_save(self):
        cali_wd = self.cali_wd
        snap_shot = self.__dict__
        with open(os.path.join(cali_wd, "GA_auto_save.pickle"), "wb") as outfile:
            # protocol=pickle.HIGHEST_PROTOCOL
            # print()
            pickle.dump(snap_shot, outfile)
        return None
