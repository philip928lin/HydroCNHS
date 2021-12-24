import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import HydroCNHS

##### Path and Load Model Test
prj_path, this_filename = os.path.split(__file__)
model_path = os.path.join(prj_path, "Calibrated_model", "Best_gwlf_abm_KGE_urbanization.yaml")
bound_path = os.path.join(prj_path, "ParBound")
wd = prj_path

model_dict = HydroCNHS.load_model(model_path)
model_dict["Path"]["WD"] = wd
model_dict["Path"]["Modules"] = os.path.join(prj_path, "ABM_modules")

with open(os.path.join(prj_path, "Inputs", "TRB_inputs.pickle"), "rb") as file:
    (temp, prec, pet, obv_D, obv_M, obv_Y) = pickle.load(file)

model_urban = HydroCNHS.Model(model_dict, "Urban")

Q_urban = model_urban.run(temp, prec, pet)
sim_Q_D_u = pd.DataFrame(Q_urban, index=model_urban.pd_date_index)
sim_Q_Y_u = sim_Q_D_u.resample("YS").mean()

#%%
best_path = os.path.join(prj_path, "Calibrated_model", "Best_gwlf_abm_KGE.yaml")
model_dict = HydroCNHS.load_model(best_path)
model_dict["Path"]["WD"] = wd
model_dict["Path"]["Modules"] = os.path.join(prj_path, "ABM_modules")
model = HydroCNHS.Model(model_dict, "Original")
Q = model.run(temp, prec, pet)
sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)
sim_Q_Y = sim_Q_D.resample("YS").mean()

fig, ax = plt.subplots()
x = np.arange(1981,2014)
ax.plot(x,sim_Q_Y_u["WSLO"], label="with urbanization")
ax.plot(x,sim_Q_Y["WSLO"], label="without urbanization", ls="dashed")
ax.set_xlim([1981, 2013])
ax.legend()
ax.set_ylabel("Streamflow ($m^3$/sec)")
ax.set_xlabel("Year")
