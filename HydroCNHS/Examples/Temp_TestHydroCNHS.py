#%%
import pandas as pd
import numpy as np
import HydroCNHS

WthData = pd.read_csv(r"C:\Users\Philip\Documents\GitHub\HydroCNHS\HydroCNHS\Examples\GWLF_TaiwanShihmenReservoir_Data.csv")
prep = WthData["P (cm)"].to_numpy() 
temp = WthData["T (degC)"].to_numpy()
ModelPath = r"C:\Users\Philip\OneDrive\Lehigh\0_Proj2_UA-SA-Equifinality\ModelRunTest\model_loadtest.yaml"
#Model = HydroCNHS.loadModel(ModelPath)


HydroCNHS.updateConfig({"Parallelization":{"Cores_formUH_Lohmann":1, "verbose":0}})
HydroCNHS.loadConfig()
Test = HydroCNHS.HydroCNHS(ModelPath, "Test")
#%%
T={}; P={}
for o in ["S1","g","S2","V11","S3","S4","V12","S5","V2","G"]:
    T[o] = temp
    P[o] = prep
    
Q = Test.run(T, P)

# 20000 runs 60yrs ~ 14 days on a single computer with single core.

#%%
Model = HydroCNHS.loadModel(ModelPath)
def toParsDFList(Sections = ["LSM","Routing","ABM"]):
    DFList = []
    for s in Sections:
        if s == "LSM":
            LSM = list(Model["LSM"].keys())
            LSM.remove("Model")
            if Model["LSM"]["Model"] == "GWLF":
                df = pd.DataFrame(index = ['CN2', 'IS', 'Res', 'Sep', 'Alpha', 'Beta', 'Ur', 'Df', 'Kc'])
            for i in LSM:
                df[i] = df.index.map(Model["LSM"][i]["Pars"])  
            DFList.append(df)
        if s == "Routing":
            Routing = list(Model["Routing"].keys())
            Routing.remove("Model")
            if Model["Routing"]["Model"] == "Lohmann":
                df = pd.DataFrame(index = ['GShape', 'GScale', 'Velo', 'Diff'])
            for end in Routing:
                for start in Model["Routing"][end]:
                    df[(start, end)] = df.index.map(Model["Routing"][end][start]["Pars"])  
            DFList.append(df)    
        if s == "ABM":
            pass        
    return DFList

a = toParsDFList()
                
            pd.concat([df, Model["LSM"][i]["Pars"]])
Routing = Model["Routing"]
RoutingOutlets = list(Routing.keys())
RoutingOutlets.remove('Model')  




























#%%
Tmax_hr = 5*24
UH_RRm = np.ones(Tmax_hr+100)

FR = np.zeros((Tmax_hr, 2))     												
FR[0:23, 0] = 1 / 24    # Later sum over 24 hours, so will need to be divided by 24.

# S-map Unit conversion, from hr to day
for t in range(Tmax_hr):
    for L in range (0, Tmax_hr+24):  
        if (t-L) > 0:   # We didn't store t = 0 (h = 0) in UH_RRm
            print(t-L)
            #print(FR[t-L,0])
            FR[t,1] = FR[t,1] + FR[t-L,0] * UH_RRm[L]

#%%
FRmatrix = np.zeros((Tmax_hr+23, Tmax_hr-1))   
for i in range(Tmax_hr-1):
    FRmatrix[:,i] = np.pad(UH_RRm, (i+1, 23), 'constant', constant_values=(0, 0))[:Tmax_hr+23]
FRmatrix = np.sum(FRmatrix, axis = 1)/24
FRmatrix1 = FRmatrix[:Tmax_hr] - np.pad(FRmatrix, (24, 0), 'constant', constant_values=(0, 0))[:Tmax_hr]
#%%
# Aggregate to daily UH
for t in range(T_RR):
    UH_RR[t] = sum(FR[(24*(t+1)-24):(24*(t+1)-1),1])