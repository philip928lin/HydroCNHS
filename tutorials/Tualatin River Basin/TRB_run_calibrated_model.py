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
best_gwlf_abm_path = os.path.join(
    prj_path, "Calibrated_model", "Best_gwlf_abm_KGE.yaml") # from RN seed 4.
best_abcd_abm_path = os.path.join(
    prj_path, "Calibrated_model", "Best_abcd_abm_KGE.yaml") # from RN seed 3.
model_dict_gwlf = HydroCNHS.load_model(best_gwlf_abm_path)
model_dict_abcd = HydroCNHS.load_model(best_abcd_abm_path)
# Change the path according to this .py file's directory.
model_dict_gwlf["Path"]["WD"] = wd
model_dict_gwlf["Path"]["Modules"] = os.path.join(prj_path, "ABM_modules")
model_dict_abcd["Path"]["WD"] = wd
model_dict_abcd["Path"]["Modules"] = os.path.join(prj_path, "ABM_modules")

##### Create HydroCNHS Model Object for Simulation.
model_gwlf = HydroCNHS.Model(model_dict_gwlf, "gwlf")
model_abcd = HydroCNHS.Model(model_dict_abcd, "abcd")

##### Run Simulation for GWLF Coupled Model
Q_gwlf = model_gwlf.run(temp, prec, pet)
sim_Q_D_gwlf = pd.DataFrame(Q_gwlf, index=model_gwlf.pd_date_index)
sim_Q_M_gwlf = sim_Q_D_gwlf.resample("MS").mean()
sim_res_M_gwlf = model_gwlf.data_collector.ResAgt["Release"]
sim_res_M_gwlf = pd.DataFrame(sim_res_M_gwlf, index=model_gwlf.pd_date_index).resample("MS").mean()
sim_div_M_gwlf = model_gwlf.data_collector.DivAgt["Diversion"]
sim_div_M_gwlf = pd.DataFrame(sim_div_M_gwlf, index=model_gwlf.pd_date_index).resample("MS").mean()

##### Run Simulation for ABCD Coupled Model
Q_abcd = model_abcd.run(temp, prec, pet)
sim_Q_D_abcd = pd.DataFrame(Q_abcd, index=model_abcd.pd_date_index)
sim_Q_M_abcd = sim_Q_D_abcd.resample("MS").mean()
sim_res_M_abcd = model_abcd.data_collector.ResAgt["Release"]
sim_res_M_abcd = pd.DataFrame(sim_res_M_abcd, index=model_abcd.pd_date_index).resample("MS").mean()
sim_div_M_abcd = model_abcd.data_collector.DivAgt["Diversion"]
sim_div_M_abcd = pd.DataFrame(sim_div_M_abcd, index=model_abcd.pd_date_index).resample("MS").mean()

#%%
##### Plot Time Series Streamflow.
fig, axes = plt.subplots(nrows=4, sharex=True)
axes = axes.flatten()
x = sim_Q_M_gwlf.index
axes[0].plot(x, sim_Q_M_gwlf["DLLO"], label="$M_{gwlf}$")
axes[1].plot(x, sim_Q_M_gwlf["WSLO"], label="$M_{gwlf}$")
axes[2].plot(x, sim_res_M_gwlf[0], label="$M_{gwlf}$")
axes[3].plot(x, sim_div_M_gwlf[0], label="$M_{gwlf}$")

axes[0].plot(x, sim_Q_M_abcd["DLLO"], ls=":", label="$M_{abcd}$")
axes[1].plot(x, sim_Q_M_abcd["WSLO"], ls=":", label="$M_{abcd}$")
axes[2].plot(x, sim_res_M_abcd[0], ls=":", label="$M_{abcd}$")
axes[3].plot(x, sim_div_M_abcd[0], ls=":", label="$M_{abcd}$")

axes[0].plot(x, obv_M["DLLO"], ls="--", lw=1, color="black", label="Obv")
axes[1].plot(x, obv_M["WSLO"], ls="--", lw=1, color="black", label="Obv")
axes[2].plot(x, obv_M["ResAgt"], ls="--", lw=1, color="black", label="Obv")
axes[3].plot(x, obv_M["DivAgt"], ls="--", lw=1, color="black", label="Obv")

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