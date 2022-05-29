import os
import HydroCNHS
prj_path, this_filename = os.path.split(__file__)

#%% ===========================================================================
# Step 1: Create a model configuration file
# =============================================================================
### Initialize a model builder object.
wd = prj_path
mb = HydroCNHS.ModelBuilder(wd)

### Setup a water system simulation information
mb.set_water_system(start_date="1981/1/1", end_date="2013/12/31")

### Setup land surface model (rainfall-runoff model)
# Here we have seven subbasins and we select GWLF as the rainfall-runoff model.
outlet_list = ['Hagg', 'DLLO', 'TRGC', 'DAIRY', 'RCTV', 'WSLO']
area_list = [10034.2408, 22568.2404, 24044.6363, 59822.7546, 19682.6046,
             47646.8477]
lat_list = [45.469, 45.475, 45.502, 45.520, 45.502, 45.350]
mb.set_rainfall_runoff(outlet_list=outlet_list,area_list=area_list,
                       lat_list=lat_list, runoff_model="GWLF")

### Setup routing 
flow_length_list = [101469.139, 91813.075, 80064.864, 70988.164, 60398.680, 0]
mb.set_routing_outlet(routing_outlet="WSLO",
                      upstream_outlet_list=outlet_list,
                      flow_length_list=flow_length_list)

### Print the model in the console
mb.print_model()

### Output initial model configuration file (.yaml) 
mb.write_model_to_yaml(filename="HydroModel.yaml")


#%% ===========================================================================
# Step 2: Populate a model configuration file
# =============================================================================

# Nothing need to be manually modified in this example.


#%% ===========================================================================
# Step 3: Run a calibration 
# =============================================================================
import matplotlib.pyplot as plt 
import pandas as pd 
import HydroCNHS.calibration as cali
from copy import deepcopy

# Load climate data
temp = pd.read_csv(os.path.join(wd,"Data","Temp_degC.csv"),
                   index_col=["Date"]).to_dict(orient="list")
prec = pd.read_csv(os.path.join(wd,"Data","Prec_cm.csv"),
                   index_col=["Date"]).to_dict(orient="list")
pet = pd.read_csv(os.path.join(wd,"Data","Pet_cm.csv"),
                   index_col=["Date"]).to_dict(orient="list")

# Load flow gauge monthly data at WSLO
obv_flow_WSLO = pd.read_csv(os.path.join(wd,"Data","WSLO_M_cms.csv"),
                            index_col=["Date"], parse_dates=["Date"])

# Load model
model_dict = HydroCNHS.load_model(os.path.join(wd, "HydroModel.yaml"))

# Generate default parameter bounds
df_list, df_name = HydroCNHS.write_model_to_df(model_dict)
par_bound_df_list, df_name = HydroCNHS.gen_default_bounds(model_dict)

# Create convertor for calibration
converter = cali.Convertor()
cali_inputs = converter.gen_cali_inputs(wd, df_list, par_bound_df_list)
formatter = converter.formatter

# Code evaluation function for GA algorthm
def evaluation(individual, info):
    cali_wd, current_generation, ith_individual, formatter, _ = info
    name = "{}-{}".format(current_generation, ith_individual)

    ##### individual -> model
    # Convert 1D array to a list of dataframes.
    df_list = cali.Convertor.to_df_list(individual, formatter)
    # Feed dataframes in df_list to model dictionary.
    model = deepcopy(model_dict)
    for i, df in enumerate(df_list):
        s = df_name[i].split("_")[0]
        model = HydroCNHS.load_df_to_model_dict(model, df, s, "Pars")

    ##### Run simuluation
    model = HydroCNHS.Model(model, name)
    Q = model.run(temp, prec, pet)

    ##### Get simulation data
    # Streamflow of routing outlets.
    cali_target = ["WSLO"]
    cali_period = ("1981-1-1", "2005-12-31")
    sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)[cali_target]
    # Resample the daily simulation output to monthly outputs.
    sim_Q_M = sim_Q_D[cali_target].resample("MS").mean()

    KGE = HydroCNHS.Indicator().KGE(
        x_obv=obv_flow_WSLO[cali_period[0]:cali_period[1]][cali_target],
        y_sim=sim_Q_M[cali_period[0]:cali_period[1]][cali_target])
    
    fitness = KGE
    return (fitness,)

config = {'min_or_max': 'max',
         'pop_size': 100,
         'num_ellite': 1,
         'prob_cross': 0.5,
         'prob_mut': 0.15,
         'stochastic': False,
         'max_gen': 100,
         'sampling_method': 'LHC',
         'drop_record': False,
         'paral_cores': -1,
         'paral_verbose': 1,
         'auto_save': True,
         'print_level': 1,
         'plot': True}

seed = 5
rn_gen = HydroCNHS.create_rn_gen(seed)
ga = cali.GA_DEAP(evaluation, rn_gen)
ga.set(cali_inputs, config, formatter, name="Cali_HydroModel_gwlf_KGE")
ga.run()
summary = ga.summary
individual = ga.solution

##### Output the calibrated model.
df_list = cali.Convertor.to_df_list(individual, formatter)
model_best = deepcopy(model_dict)
for i, df in enumerate(df_list):
    s = df_name[i].split("_")[0]
    model = HydroCNHS.load_df_to_model_dict(model_best, df, s, "Pars")
HydroCNHS.write_model(model_best, os.path.join(ga.cali_wd, "Best_HydroModel_gwlf_KGE.yaml"))

#%% ===========================================================================
# Step 3: Run a simulation 
# =============================================================================
### Run a simulation.
model = HydroCNHS.Model(os.path.join(wd, "Cali_HydroModel_gwlf_KGE",
                                     "Best_HydroModel_gwlf_KGE.yaml"))
Q = model.run(temp, prec, pet)
result = pd.DataFrame(Q, index=model.pd_date_index).resample("MS").mean()

### Plot
fig, ax = plt.subplots()
ax.plot(obv_flow_WSLO.index, obv_flow_WSLO.loc[:, "WSLO"], label="Obv")
ax.plot(obv_flow_WSLO.index, result["WSLO"], ls="--", label="Sim")
ax.legend()
















