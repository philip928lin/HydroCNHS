import os
import pandas as pd
import matplotlib.pyplot as plt
import HydroCNHS
prj_path, this_filename = os.path.split(__file__)

wd = prj_path

# Load climate data
temp = pd.read_csv(os.path.join(wd,"Data","Temp_degC.csv"),
                   index_col=["Date"]).to_dict(orient="list")
prec = pd.read_csv(os.path.join(wd,"Data","Prec_cm.csv"),
                   index_col=["Date"]).to_dict(orient="list")
pet = pd.read_csv(os.path.join(wd,"Data","Pet_cm.csv"),
                   index_col=["Date"]).to_dict(orient="list")

# Load flow gauge monthly data
obv_M = pd.read_csv(os.path.join(wd,"Data","Cali_M_cms.csv"),
                    index_col=["Date"], parse_dates=["Date"])

filename = os.path.join(wd, "Cali_gwlf_abm_KGE_5", "Calibrated_TRB_GWLF.yaml")
model = HydroCNHS.Model(filename)
Q = model.run(temp, prec, pet)
sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)[["WSLO","DLLO"]]
sim_Q_D["ResAgt"] = model.dc.ResAgt["Release"]
sim_Q_D["DivAgt"] = model.dc.DivAgt["Diversion"]
sim_Q_M_gwlf = sim_Q_D[["WSLO","DLLO","ResAgt","DivAgt"]].resample("MS").mean()

filename = os.path.join(wd, "Cali_abcd_abm_KGE_10", "Calibrated_TRB_ABCD.yaml")
model = HydroCNHS.Model(filename)
Q = model.run(temp, prec, pet)
sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)[["WSLO","DLLO"]]
sim_Q_D["ResAgt"] = model.dc.ResAgt["Release"]
sim_Q_D["DivAgt"] = model.dc.DivAgt["Diversion"]
sim_Q_M_abcd = sim_Q_D[["WSLO","DLLO","ResAgt","DivAgt"]].resample("MS").mean()

#%%
##### Plot Time Series Streamflow.
fig, axes = plt.subplots(nrows=4, sharex=True)
axes = axes.flatten()
x = sim_Q_M_gwlf.index
axes[0].plot(x, sim_Q_M_gwlf["DLLO"], label="$M_{gwlf}$")
axes[1].plot(x, sim_Q_M_gwlf["WSLO"], label="$M_{gwlf}$")
axes[2].plot(x, sim_Q_M_gwlf["ResAgt"], label="$M_{gwlf}$")
axes[3].plot(x, sim_Q_M_gwlf["DivAgt"], label="$M_{gwlf}$")

axes[0].plot(x, sim_Q_M_abcd["DLLO"], ls=":", label="$M_{abcd}$")
axes[1].plot(x, sim_Q_M_abcd["WSLO"], ls=":", label="$M_{abcd}$")
axes[2].plot(x, sim_Q_M_abcd["ResAgt"], ls=":", label="$M_{abcd}$")
axes[3].plot(x, sim_Q_M_abcd["DivAgt"], ls=":", label="$M_{abcd}$")

axes[0].plot(x, obv_M["DLLO"], ls="--", lw=1, color="black", label="Obv")
axes[1].plot(x, obv_M["WSLO"], ls="--", lw=1, color="black", label="Obv")
axes[2].plot(x, obv_M["ResAgt"], ls="--", lw=1, color="black", label="Obv")
axes[3].plot(x, obv_M["DivAgt"], ls="--", lw=1, color="black", label="Obv")

axes[0].set_ylim([0,75])
axes[1].set_ylim([0,290])
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