import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import HydroCNHS

##### Setup Working Directory
# Get this .py file's directory.
prj_path, this_filename = os.path.split(__file__)
wd = prj_path

##### Load Daily Weather Time Series.
with open(os.path.join(prj_path, "Inputs", "TRB_inputs.pickle"), "rb") as file:
    (temp, prec, pet, obv_D, obv_M, obv_Y) = pickle.load(file)
    
##### Load Model.yaml.
best_gwlf_abm_path = os.path.join(prj_path, "Calibrated_model", "Best_gwlf_abm_KGE.yaml")
model_dict_gwlf = HydroCNHS.load_model(best_gwlf_abm_path)
# Change the path according to this .py file's directory.
model_dict_gwlf["Path"]["WD"] = wd
model_dict_gwlf["Path"]["Modules"] = os.path.join(prj_path, "ABM_modules")

##### Create HydroCNHS Model Object for Simulation.
model_gwlf = HydroCNHS.Model(model_dict_gwlf, "gwlf")

##### Run simulation
Q_gwlf = model_gwlf.run(temp, prec, pet) # pet is optional.



sim_Q_D_gwlf = pd.DataFrame(Q_gwlf, index=model_gwlf.pd_date_index)
sim_Q_M_gwlf = sim_Q_D_gwlf.resample("MS").mean()
sim_res_M_gwlf = model_gwlf.data_collector.R1["release"]
sim_res_M_gwlf = pd.DataFrame(sim_res_M_gwlf, index=model_gwlf.pd_date_index).resample("MS").mean()
sim_div_M_gwlf = model_gwlf.data_collector.SHPP["Div"]
sim_div_M_gwlf = pd.DataFrame(sim_div_M_gwlf, index=model_gwlf.pd_date_index).resample("MS").mean()

fig, axes = plt.subplots(nrows=4, sharex=True)
axes = axes.flatten()
x = sim_Q_M_gwlf.index
axes[0].plot(x, sim_Q_M_gwlf["DLLO"], label="$M_{gwlf}$")
axes[1].plot(x, sim_Q_M_gwlf["WSLO"], label="$M_{gwlf}$")
axes[2].plot(x, sim_res_M_gwlf[0], label="$M_{gwlf}$")
axes[3].plot(x, sim_div_M_gwlf[0], label="$M_{gwlf}$")

axes[0].plot(x, obv_M["DLLO"], ls="--", lw=1, color="black", label="Obv")
axes[1].plot(x, obv_M["WSLO"], ls="--", lw=1, color="black", label="Obv")
axes[2].plot(x, obv_M["SCOO"], ls="--", lw=1, color="black", label="Obv")
axes[3].plot(x, obv_M["SHPP"], ls="--", lw=1, color="black", label="Obv")

axes[0].set_ylim([0,75])
axes[1].set_ylim([0,230])
axes[2].set_ylim([0,23])
axes[3].set_ylim([0,2])

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