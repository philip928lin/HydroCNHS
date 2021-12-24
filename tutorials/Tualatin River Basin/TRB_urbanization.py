import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import HydroCNHS

##### Path and Load Model Test
pc = "Philip"
prj_path = r"C:\Users\{}\OneDrive\Lehigh\0_Proj2_HydroCNHS".format(pc)
model_path = os.path.join(prj_path, "Model", "Best_gwlf_abm_KGE_urbanization.yaml")
data_path = os.path.join(prj_path, "Data")
bound_path = os.path.join(prj_path, "Model", "ParBound")
wd = r"C:\Users\{}\Documents\TRB".format(pc)
model_dict = HydroCNHS.load_model(model_path)


for k in model_dict["Path"]:
    model_dict["Path"][k] = os.path.join(prj_path, "Model")

with open(os.path.join(prj_path, "Model", "TRB_inputs.pickle"), "rb") as file:
    (temp, prec, pet, obv_D, obv_M, obv_Y) = pickle.load(file)

model_urban = HydroCNHS.Model(model_dict, "Urban")

Q_urban = model_urban.run(temp, prec, pet)
sim_Q_D_u = pd.DataFrame(Q_urban, index=model_urban.pd_date_index)
sim_Q_Y_u = sim_Q_D_u.resample("YS").mean()

#%%
best_path = os.path.join(prj_path, "Model", "Best_gwlf_abm_KGE.yaml")
model = HydroCNHS.Model(best_path, "Original")
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



#%%
# Test random seed~
rn_gen = HydroCNHS.create_rn_gen(9)
model_urban = HydroCNHS.Model(model_dict, "Urban", rn_gen)
rnn = model_urban.data_collector.U_RCTV