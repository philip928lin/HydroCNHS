Calibration
================

Genetic algorithm calibration code outline
-------------------------------

.. code-block:: python

    import HydroCNHS
    import HydroCNHS.calibration as cali

    # Assisting functions
    cali.helper()
    cali.get_config_template()
    cali.get_inputs_template()

    def evaluation(individual, info):
        (cali_wd, current_generation, ith_individual,
        formatter, random_generator) = info
        # Run model.
        # Calculate fitness. 
        # E.g.,
        fitness = sum(individual)
        return (fitness,)

    config = {'min_or_max': 'max',
            'pop_size': 200,
            'num_ellite': 1,
            'prob_cross': 0.5,
            'prob_mut': 0.1,
            'stochastic': False,
            'max_gen': 100,
            'sampling_method': 'LHC',
            'drop_record': False,
            'paral_cores': -1,
            'paral_verbose': 1,
            'auto_save': True,
            'print_level': 1,
            'plot': True}

    cali_inputs = {"par_name":    ["a","b","c"],     
                "par_bound":   [[1,2],[1,2],[1,2]],      
                "wd":          "working directory"} 

    seed = 3
    rn_gen = HydroCNHS.create_rn_gen(seed)  # Optional
    ga = cali.GA_DEAP(evaluation, rn_gen)
    ga.set(cali_inputs, config, name="Cali_example")
    ga.run()
    ga.run_individual(ga.solution)  # Output performance (.txt) of solution.
    summary = ga.summary
    print(summary)



Converter
---------




TRB calibration example
-----------------------