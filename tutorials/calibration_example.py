import os
import HydroCNHS
import HydroCNHS.calibration as cali

prj_path, this_filename = os.path.split(__file__)


def evaluation(individual, info):
    x = individual
    fitness = -x[0] ** 2 + 5 * x[0] - x[1]
    return (fitness,)

    cali_wd, current_generation, ith_individual, formatter, _ = info
    name = "{}-{}".format(current_generation, ith_individual)

    ##### individual -> model
    # Convert 1D array to a list of dataframes.
    df_list = cali.Convertor.to_df_list(individual, formatter)
    # Feed dataframes in df_list to model dictionary.


# Calibration inputs
cali.get_inputs_template()  # print an input template.

inputs = {
    "par_name": ["x1", "x2"],
    "par_bound": [[-5, 5], [-5, 5]],
    "wd": "working directory",
}
inputs["wd"] = prj_path

# GA configuration
cali.get_config_template()

config = {
    "min_or_max": "max",  # maximize or minimize the evaluation function.
    "pop_size": 100,  # Size of the population.
    "num_ellite": 1,  # Number of ellites.
    "prob_cross": 0.5,  # Crossover probability for uniform crossover operator.
    "prob_mut": 0.15,  # Mutation probability of each parameter.
    "stochastic": False,  # Is the evaluation stochastic?
    "max_gen": 100,  # Maximum generation number.
    "sampling_method": "LHC",  # Sampling method for the initial population.
    "drop_record": False,  # Whether to drop historical records to save space.
    "paral_cores": -1,  # Number of parallel cores. -1 means all available cores.
    "paral_verbose": 1,  # Higher value will output more console messages.
    "auto_save": True,  # If true, users may continue the run later on by loading the auto-save file.
    "print_level": 1,  # Control the number of generations before the printing summary of GA run.
    "plot": True,
}  # Plot to time series of the best fitnesses over a generation.

# Run GA
rn_gen = HydroCNHS.create_rn_gen(seed=3)
ga = cali.GA_DEAP(evaluation, rn_gen)
ga.set(inputs, config, formatter=None, name="Cali_example")
ga.run()
ga.solution
# Out[0]: array([ 2.47745344, -4.96991833])

# %%
# Continue the run with larger "max_gen"
ga = cali.GA_DEAP(evaluation, rn_gen)
ga.load(os.path.join(prj_path, "Cali_example", "GA_auto_save.pickle"), max_gen=120)
ga.run()
# =====Generation 120=====
#   Elapsed time 00:00:05
#   Min -6.69464
#   Max 11.21948
#   Avg 10.99626
#   Std 1.77931

# GA done!
