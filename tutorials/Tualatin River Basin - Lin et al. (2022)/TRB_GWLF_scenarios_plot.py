import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import HydroCNHS

prj_path, this_filename = os.path.split(__file__)

wd = prj_path

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
prec_M = (
    pd.read_csv(
        os.path.join(wd, "Data", "Pet_cm.csv"), index_col=["Date"], parse_dates=["Date"]
    )
    .resample("MS")
    .mean()
)

folder = "Cali_gwlf_abm_KGE_5"
filename = os.path.join(wd, folder, "Calibrated_TRB_GWLF.yaml")
model = HydroCNHS.load_model(filename)
model["Path"]["WD"] = wd
model["Path"]["Modules"] = wd

# Urbanization scenario
model_urban = deepcopy(model)
model_urban["WaterSystem"]["ABM"]["InsituAPI"].append("Drain_AgtType")
model_urban["ABM"]["Drain_AgtType"] = {
    "DrainAgt1": {
        "Attributes": {},
        "Inputs": {"Priority": 1, "Links": {"RCTV": 1}, "DMClass": None},
        "Pars": {},
    },
    "DrainAgt2": {
        "Attributes": {},
        "Inputs": {"Priority": 1, "Links": {"WSLO": 1}, "DMClass": None},
        "Pars": {},
    },
}

HydroCNHS.write_model(
    model_urban, os.path.join(wd, folder, "Calibrated_TRB_GWLF_urban.yaml")
)

# Fixed diversion scenario
model_fixed_div = deepcopy(model)
model_fixed_div["WaterSystem"]["ABM"]["DMClasses"] = [
    "ReleaseDM",
    "TransferDM",
    "FixedDivertDM",
]
model_fixed_div["WaterSystem"]["ABM"]["RiverDivAPI"] = ["FixedDiversion_AgtType"]
rf = model_fixed_div["ABM"]["Diversion_AgtType"]["DivAgt"]["Pars"]["ReturnFactor"]
model_fixed_div["ABM"]["FixedDiversion_AgtType"] = {
    "DivAgt": {
        "Attributes": {},
        "Inputs": {
            "Priority": 1,
            "Links": {"TRGC": -1, "WSLO": ["ReturnFactor", 0, "Plus"]},
            "DMClass": "FixedDivertDM",
        },
        "Pars": {"ReturnFactor": rf},
    }
}
model_fixed_div["ABM"].pop("Diversion_AgtType")

HydroCNHS.write_model(
    model_fixed_div, os.path.join(wd, folder, "Calibrated_TRB_GWLF_fixedDiv.yaml")
)
# %%
##### Urbanization scenario
filename = os.path.join(wd, folder, "Calibrated_TRB_GWLF.yaml")
model = HydroCNHS.load_model(filename)
model["Path"]["WD"] = wd
model["Path"]["Modules"] = wd
model = HydroCNHS.Model(model)
Q = model.run(temp, prec, pet)
sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)[["WSLO", "DLLO"]]
sim_Q_D["ResAgt"] = model.dc.ResAgt["Release"]
sim_Q_D["DivAgt"] = model.dc.DivAgt["Diversion"]
sim_Q_Y = sim_Q_D[["WSLO", "DLLO", "ResAgt", "DivAgt"]].resample("YS").mean()

filename = os.path.join(wd, folder, "Calibrated_TRB_GWLF_urban.yaml")
model = HydroCNHS.load_model(filename)
model["Path"]["WD"] = wd
model["Path"]["Modules"] = wd
model = HydroCNHS.Model(model)
Q = model.run(temp, prec, pet)
sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)[["WSLO", "DLLO"]]
sim_Q_D["ResAgt"] = model.dc.ResAgt["Release"]
sim_Q_D["DivAgt"] = model.dc.DivAgt["Diversion"]
sim_Q_Y_urban = sim_Q_D[["WSLO", "DLLO", "ResAgt", "DivAgt"]].resample("YS").mean()

##### Plot Time Series Streamflow.
fig, ax = plt.subplots()
x = np.arange(1981, 2014)
ax.plot(x, sim_Q_Y_urban["WSLO"], label="with urbanization")
ax.plot(x, sim_Q_Y["WSLO"], label="without urbanization", ls="dashed")
ax.set_xlim([1981, 2013])
ax.legend()
ax.set_ylabel("Streamflow ($m^3$/sec)")
ax.set_xlabel("Year")


# %%
##### Fixed diversion scenario
filename = os.path.join(wd, folder, "Calibrated_TRB_GWLF.yaml")
model = HydroCNHS.load_model(filename)
model["Path"]["WD"] = wd
model["Path"]["Modules"] = wd
model = HydroCNHS.Model(model)
Q = model.run(temp, prec, pet)
sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)[["WSLO", "DLLO"]]
sim_Q_D["ResAgt"] = model.dc.ResAgt["Release"]
sim_Q_D["DivAgt"] = model.dc.DivAgt["Diversion"]
sim_Q_M = sim_Q_D[["WSLO", "DLLO", "ResAgt", "DivAgt"]].resample("MS").mean()
sim_Q_Y = sim_Q_D[["WSLO", "DLLO", "ResAgt", "DivAgt"]].resample("YS").mean()
mask = [i.month in [6, 7, 8, 9] for i in sim_Q_M.index]
sim_Y = sim_Q_M.loc[mask, :].resample("YS").mean()

filename = os.path.join(wd, folder, "Calibrated_TRB_GWLF_fixedDiv.yaml")
model = HydroCNHS.load_model(filename)
model["Path"]["WD"] = wd
model["Path"]["Modules"] = wd
model = HydroCNHS.Model(model)
Q = model.run(temp, prec, pet)
sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)[["WSLO", "DLLO"]]
sim_Q_D["ResAgt"] = model.dc.ResAgt["Release"]
sim_Q_D["DivAgt"] = model.dc.DivAgt["Diversion"]
sim_Q_M_FixedDiv = sim_Q_D[["WSLO", "DLLO", "ResAgt", "DivAgt"]].resample("MS").mean()
sim_Q_Y_FixedDiv = sim_Q_D[["WSLO", "DLLO", "ResAgt", "DivAgt"]].resample("YS").mean()
mask = [i.month in [6, 7, 8, 9] for i in sim_Q_M_FixedDiv.index]
sim_Y_FixedDiv = sim_Q_M_FixedDiv.loc[mask, :].resample("YS").mean()

df = (sim_Y.std() - sim_Y_FixedDiv.std())[["ResAgt", "DLLO", "DivAgt", "WSLO"]]

##### Plot barplot
ax = df.plot(kind="bar", color=(df > 0).map({True: "g", False: "r"}))
ax.axhline(0, color="k", lw=0.5)
ax.tick_params(axis="x", labelrotation=0)
ax.set_ylabel("($M_{gwlf,endog} - M_{gwlf,fixed} (m^3/sec$)")


# %% Plot two scenarios together
fig, axes = plt.subplots(ncols=2)
plt.tight_layout()
axes = axes.flatten()
ax = axes[0]
df.plot(kind="bar", color=(df > 0).map({True: "g", False: "r"}), ax=ax)
ax.axhline(0, color="k", lw=0.5)
ax.tick_params(axis="x", labelrotation=0)
ax.set_ylabel("($M_{gwlf,endog} - M_{gwlf,fixed} (m^3/sec$)")

ax = axes[1]
x = np.arange(1981, 2014)
ax.plot(x, sim_Q_Y_urban["WSLO"], label="with urbanization")
ax.plot(x, sim_Q_Y["WSLO"], label="without urbanization", ls="dashed")
ax.set_xlim([1981, 2013])
ax.legend()
ax.set_ylabel("Streamflow ($m^3$/sec)")
ax.set_xlabel("Year")
