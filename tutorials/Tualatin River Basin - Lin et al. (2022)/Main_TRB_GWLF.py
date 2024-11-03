import os
import HydroCNHS

prj_path, this_filename = os.path.split(__file__)
# %% ===========================================================================
# Step 1: Create a model configuration file
# =============================================================================
### Initialize a model builder object.
wd = prj_path
mb = HydroCNHS.ModelBuilder(wd)

### Setup a water system simulation information
mb.set_water_system(start_date="1981/1/1", end_date="2013/12/31")

### Setup land surface model (rainfall-runoff model)
# Here we have seven subbasins and we select GWLF as the rainfall-runoff model.
outlet_list = ["HaggIn", "TRTR", "DLLO", "TRGC", "DAIRY", "RCTV", "WSLO"]
area_list = [
    10034.2408,
    329.8013,
    22238.4391,
    24044.6363,
    59822.7546,
    19682.6046,
    47646.8477,
]
lat_list = [45.469, 45.458, 45.475, 45.502, 45.520, 45.502, 45.350]
mb.set_rainfall_runoff(
    outlet_list=outlet_list, area_list=area_list, lat_list=lat_list, runoff_model="GWLF"
)

### Setup routing outlets
# Add WSLO
mb.set_routing_outlet(
    routing_outlet="WSLO",
    upstream_outlet_list=["TRGC", "DAIRY", "RCTV", "WSLO"],
    flow_length_list=[80064.864, 70988.164, 60398.680, 0],
)
# Add TRGC
mb.set_routing_outlet(
    routing_outlet="TRGC",
    upstream_outlet_list=["DLLO", "TRGC"],
    flow_length_list=[11748.211, 0],
)
# Add DLLO
# Specify that ResAgt is an instream object.
mb.set_routing_outlet(
    routing_outlet="DLLO",
    upstream_outlet_list=["ResAgt", "TRTR", "DLLO"],
    flow_length_list=[9656.064, 30899.4048, 0],
    instream_objects=["ResAgt"],
)
# Add HaggIn
mb.set_routing_outlet(
    routing_outlet="HaggIn", upstream_outlet_list=["HaggIn"], flow_length_list=[0]
)

### Setup ABM
mb.set_ABM(abm_module_folder_path=wd, abm_module_name="TRB_ABM.py")
mb.add_agent(
    agt_type_class="Reservoir_AgtType",
    agt_name="ResAgt",
    api=mb.api.Dam,
    link_dict={"HaggIn": -1, "ResAgt": 1},
    dm_class="ReleaseDM",
)
mb.add_agent(
    agt_type_class="Diversion_AgtType",
    agt_name="DivAgt",
    api=mb.api.RiverDiv,
    link_dict={"TRGC": -1, "WSLO": ["ReturnFactor", 0, "Plus"]},
    dm_class="DivertDM",
    par_dict={"ReturnFactor": [-99], "a": -99, "b": -99},
)
mb.add_agent(
    agt_type_class="Pipe_AgtType",
    agt_name="PipeAgt",
    api=mb.api.Conveying,
    link_dict={"TRTR": 1},
    dm_class="TransferDM",
)

### Print the model in the console
mb.print_model()

### Output initial model configuration file (.yaml) and ABM module template.
mb.write_model_to_yaml(filename="TRB_GWLF.yaml")
mb.gen_ABM_module_template()

# %% ===========================================================================
# Step 2: Populate a model configuration file
# =============================================================================

# Nothing need to be manually modified in this example.

# %% ===========================================================================
# Step 3: Program ABM module (.py)
# =============================================================================

# We provide a complete ABM module for the TRB example (i.e.,
# TRB_ABM_complete.py). Theorical details can be found in Lin et al. (2022).

# %% ===========================================================================
# Step 4: Run a calibration
# =============================================================================
import matplotlib.pyplot as plt
import pandas as pd
import HydroCNHS.calibration as cali
from copy import deepcopy

# Load climate data
temp = pd.read_csv(
    os.path.join(wd, "Data", "Temp_degC.csv"), index_col=["Date"]
).to_dict(orient="list")
prec = pd.read_csv(os.path.join(wd, "Data", "Prec_cm.csv"), index_col=["Date"]).to_dict(
    orient="list"
)
pet = pd.read_csv(os.path.join(wd, "Data", "Pet_cm.csv"), index_col=["Date"]).to_dict(
    orient="list"
)

# Load flow gauge monthly data
obv_M = pd.read_csv(
    os.path.join(wd, "Data", "Cali_M_cms.csv"), index_col=["Date"], parse_dates=["Date"]
)

# Load model
model_dict = HydroCNHS.load_model(os.path.join(wd, "TRB_GWLF.yaml"))
# Change the ABM module to the complete one.
model_dict["WaterSystem"]["ABM"]["Modules"] = ["TRB_ABM_complete.py"]

# Generate default parameter bounds
df_list, df_name = HydroCNHS.write_model_to_df(model_dict)
par_bound_df_list, df_name = HydroCNHS.gen_default_bounds(model_dict)

# Modify the default bounds of ABM
df_abm_bound = par_bound_df_list[2]
df_abm_bound.loc["ReturnFactor.0", [("DivAgt", "Diversion_AgtType")]] = "[0, 0.5]"
df_abm_bound.loc["a", [("DivAgt", "Diversion_AgtType")]] = "[0, 3]"
df_abm_bound.loc["b", [("DivAgt", "Diversion_AgtType")]] = "[-2, 2]"

# Create convertor for calibration
converter = cali.Convertor()
cali_inputs = converter.gen_cali_inputs(wd, df_list, par_bound_df_list)
formatter = converter.formatter


# Code evaluation function for GA algorthm
def cal_batch_indicator(period, target, df_obv, df_sim):
    """Compute mean indicator over targets"""
    df_obv = df_obv[period[0] : period[1]]
    df_sim = df_sim[period[0] : period[1]]
    Indicator = HydroCNHS.Indicator()
    df = pd.DataFrame()
    for item in target:
        df_i = Indicator.cal_indicator_df(df_obv[item], df_sim[item], index_name=item)
        df = pd.concat([df, df_i], axis=0)
    df_mean = pd.DataFrame(df.mean(axis=0), columns=["Mean"]).T
    df = pd.concat([df, df_mean], axis=0)
    return df


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
    cali_target = ["WSLO", "DLLO", "ResAgt", "DivAgt"]
    cali_period = ("1981-1-1", "2005-12-31")
    vali_period = ("2006-1-1", "2013-12-31")
    sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)[["WSLO", "DLLO"]]
    sim_Q_D["ResAgt"] = model.dc.ResAgt["Release"]
    sim_Q_D["DivAgt"] = model.dc.DivAgt["Diversion"]
    # Resample the daily simulation output to monthly outputs.
    sim_Q_M = sim_Q_D[cali_target].resample("MS").mean()

    ##### Calculate indicators
    df_cali_Q_M = cal_batch_indicator(cali_period, cali_target, obv_M, sim_Q_M)
    df_vali_Q_M = cal_batch_indicator(vali_period, cali_target, obv_M, sim_Q_M)

    ##### Save output.txt
    # Only exercute when ga.run_individual(ga.solution)
    if current_generation == "best":
        with open(os.path.join(cali_wd, "cali_indiv_" + name + ".txt"), "w") as f:
            f.write("\n\nMonthly cali/vali result\n")
            f.write(df_cali_Q_M.round(3).to_csv(sep="\t").replace("\n", ""))
            f.write("\n")
            f.write(df_vali_Q_M.round(3).to_csv(sep="\t").replace("\n", ""))
            f.write("\n=========================================================\n")
            f.write("Sol:\n")
            df = pd.DataFrame(individual, index=cali_inputs["par_name"]).round(4)
            f.write(df.to_string(header=False, index=True))

    fitness = df_cali_Q_M.loc["Mean", "KGE"]
    return (fitness,)


config = {
    "min_or_max": "max",
    "pop_size": 200,
    "num_ellite": 1,
    "prob_cross": 0.5,
    "prob_mut": 0.1,
    "stochastic": False,
    "max_gen": 100,
    "sampling_method": "LHC",
    "drop_record": False,
    "paral_cores": -1,
    "paral_verbose": 1,
    "auto_save": True,
    "print_level": 1,
    "plot": True,
}

# Calibrate with 3 seeds.
seeds = [5, 10, 13]
for seed in seeds:
    rn_gen = HydroCNHS.create_rn_gen(seed)
    ga = cali.GA_DEAP(evaluation, rn_gen)
    ga.set(cali_inputs, config, formatter, name="Cali_gwlf_abm_KGE_{}".format(seed))
    ga.run()
    ga.run_individual(ga.solution)  # Output performance (.txt) of solution.

    ##### Output the calibrated model.
    individual = ga.solution
    df_list = cali.Convertor.to_df_list(individual, formatter)
    model_best = deepcopy(model_dict)
    for i, df in enumerate(df_list):
        s = df_name[i].split("_")[0]
        model = HydroCNHS.load_df_to_model_dict(model_best, df, s, "Pars")
    HydroCNHS.write_model(
        model_best, os.path.join(ga.cali_wd, "Calibrated_TRB_GWLF.yaml")
    )
    summary = ga.summary
# %% ===========================================================================
# Step 5: Run a simulation
# =============================================================================
### Run a simulation.
folder = "Cali_gwlf_abm_KGE_5"
filename = os.path.join(wd, folder, "Calibrated_TRB_GWLF.yaml")
model = HydroCNHS.Model(os.path.join(filename))
Q = model.run(temp, prec, pet)
sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)[["WSLO", "DLLO"]]
sim_Q_D["ResAgt"] = model.dc.ResAgt["Release"]
sim_Q_D["DivAgt"] = model.dc.DivAgt["Diversion"]
sim_Q_M = sim_Q_D[["WSLO", "DLLO", "ResAgt", "DivAgt"]].resample("MS").mean()
### Plot
fig, axes = plt.subplots(nrows=4, sharex=True)
axes = axes.flatten()
x = sim_Q_M.index
axes[0].plot(x, sim_Q_M["DLLO"], label="$M_{gwlf}$")
axes[1].plot(x, sim_Q_M["WSLO"], label="$M_{gwlf}$")
axes[2].plot(x, sim_Q_M["ResAgt"], label="$M_{gwlf}$")
axes[3].plot(x, sim_Q_M["DivAgt"], label="$M_{gwlf}$")

axes[0].plot(x, obv_M["DLLO"], ls="--", lw=1, color="black", label="Obv")
axes[1].plot(x, obv_M["WSLO"], ls="--", lw=1, color="black", label="Obv")
axes[2].plot(x, obv_M["ResAgt"], ls="--", lw=1, color="black", label="Obv")
axes[3].plot(x, obv_M["DivAgt"], ls="--", lw=1, color="black", label="Obv")

axes[0].set_ylim([0, 75])
axes[1].set_ylim([0, 230])
axes[2].set_ylim([0, 23])
axes[3].set_ylim([0, 2])

axes[0].set_ylabel("DLLO\n($m^3/s$)")
axes[1].set_ylabel("WSLO\n($m^3/s$)")
axes[2].set_ylabel("Release\n($m^3/s$)")
axes[3].set_ylabel("Diversion\n($m^3/s$)")

axes[0].axvline(pd.to_datetime("2006-1-1"), color="grey", ls="-", lw=1)
axes[1].axvline(pd.to_datetime("2006-1-1"), color="grey", ls="-", lw=1)
axes[2].axvline(pd.to_datetime("2006-1-1"), color="grey", ls="-", lw=1)
axes[3].axvline(pd.to_datetime("2006-1-1"), color="grey", ls="-", lw=1)

axes[0].legend(ncol=3, bbox_to_anchor=(1, 1.5), fontsize=9)

fig.align_ylabels(axes)
