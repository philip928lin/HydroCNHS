# %% Import packages and data
import pandas as pd
import HydroCNHS
from pprint import pprint
from HydroCNHS.LSM import runGWLF, calPEt_Hamon
WthData = pd.read_csv("GWLF_TaiwanShihmenReservoir_Data.csv")
P = WthData["P (cm)"].to_numpy() 
T = WthData["T (degC)"].to_numpy()

ModelPath = "GWLF_TaiwanShihmenReservoir_model.yaml"
model = HydroCNHS.loadModel(ModelPath)
pprint(model)

# %% Calculate PE
Lat = model["LSM"]["Shihmen"]["Inputs"]["Latitude"]
StartDate = model["WaterSystem"]["StartDate"]
PE = calPEt_Hamon(T, Lat, StartDate)

# %% run GWLF
GWLFPars = model["LSM"]["Shihmen"]["Pars"]
Inputs = model["LSM"]["Shihmen"]["Inputs"]
DataLength = model["WaterSystem"]["DataLength"]
Q = runGWLF(GWLFPars, Inputs, T, P, PE, StartDate, DataLength)

# %% Show (Daily)
StartDate = pd.to_datetime(StartDate, format="%Y/%m/%d")                               # to Datetime
pdDatedateIndex = pd.date_range(start = StartDate, periods = DataLength, freq = "D")   # gen pd dateIndex

Result = pd.DataFrame()
Result["SimQ [cms]"] = Q
Result["ObvQ [cms]"] = WthData["ObvQ (cms)"].to_numpy()
Result.index = pdDatedateIndex

Result.plot()
print("Correlation of daily result: ", Result.corr().iloc[0,1])

# %% Show (Monthly)
ResultM = Result.resample("M").mean()
ResultM.plot()
print("Correlation of monthly result: ", ResultM.corr().iloc[0,1])

# %% Show (Yearly)
ResultY = Result.resample("Y").mean()
ResultY.plot()
print("Correlation of yearly result: ", ResultY.corr().iloc[0,1])
