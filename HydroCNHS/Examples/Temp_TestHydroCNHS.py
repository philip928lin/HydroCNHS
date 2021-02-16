#%%
import pandas as pd
import HydroCNHS

WthData = pd.read_csv(r"C:\Users\Philip\Documents\GitHub\HydroCNHS\HydroCNHS\Examples\GWLF_TaiwanShihmenReservoir_Data.csv")
prep = WthData["P (cm)"].to_numpy() 
temp = WthData["T (degC)"].to_numpy()
ModelPath = r"C:\Users\Philip\OneDrive\Lehigh\0_Proj2_UA-SA-Equifinality\ModelRunTest\model_loadtest.yaml"
#Model = HydroCNHS.loadModel(ModelPath)

Test = HydroCNHS.HydroCNHS(ModelPath, "Test")
Test2 = HydroCNHS.HydroCNHS(ModelPath, "Test")
#%%
T={}; P={}
for o in ["S1","g","S2","V11","S3","S4","V12","S5","V2","G"]:
    T[o] = temp
    P[o] = prep
    
Q = Test.run(T, P)
Q2 = Test2.run(T, P)
# %% Calculate PE



