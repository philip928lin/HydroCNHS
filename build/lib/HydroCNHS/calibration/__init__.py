from .convertor import Convertor
from .ga_deap import GA_DEAP
def helper():
    print("(1) Code your customize evaluation function for GA algorithm with",
          "certain arguments.\nEx:\ndef evaluation(individual, formatter,",
          "info):\n\t# info = (cali_wd, current_generation, ith_individual)",
          "\n\tRun model\n\tCalculate fitness\n\n\treturn (fitness,)\n\n(2)",
          "Create calibration object from GA_DEAP() class",
          "given inputs and config.\n\n(3) Run GA.")

def get_config_template():
    config = {
        "min_or_max":         "min",
        "pop_size":           100,    
        "num_ellite":          1,     
        "prob_cross":          0.8,  
        "prob_mut":            0.1,   
        "stochastic":         False,  
        "max_gen":             100,    
        "sampling_method":     "LHC",  
        "drop_record":         False,  
        "paral_cores":         -2, 
        "paral_verbose":        0,
        "auto_save":           True,   
        "print_level":         10,      
        "plot":               True    
        }
    return config

def get_inputs_template():
    inputs = {
        "par_name":    ["a","b","c"],     
        "par_bound":   [[1,2],[1,2],[1,2]],      
        "wd":          "working directory"} 
    print("\nNote:\n Converter() can assist users",
          "to get inputs and formattor that can convert individual",
          "(1D array) back to Dataframe.")
    return inputs