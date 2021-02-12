#%%
# Demonstration for DMCGA using Himmelblau's function, which has 4 local optimal values = 0.
# Description of Himmelblau's function: https://en.wikipedia.org/wiki/Test_functions_for_optimization
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# 2021/02/11

from HydroCNHS.DMCGA import DMCGA
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

# Define loss function
def HimmelblauFunc(var, Formatter = None, SubWDInfo = None):
    # -5 <= x,y <= 5
    x = var[0]
    y = var[1]
    return (x**2+y-11)**2 + (x+y**2-7)**2

r"""
# Four optimal solutions
HimmelblauFunc([3,2])                   = 0
HimmelblauFunc([-2.805118,3.131312])    = 0
HimmelblauFunc([-3.779310,-3.283186])   = 0
HimmelblauFunc([3.584428,-1.848126])    = 0
"""
# Plot HimmelblauFunc
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = HimmelblauFunc([X, Y])
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='nipy_spectral', edgecolor='none')
ax.view_init(70, 15)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Himmelblau's Function");

# %%
# Initialize DMCGA
Inputs = {"ParName":["x", "y"], 
          "ParBound":[[-5, 5], [-5, 5]],  # [upper, low] or [4, 6, 9] Even for category type, it has to be numbers!
          "ParType":["real","real"],   # real or category
          "ParWeight":[1, 1],  
          "WD":r"C:\Users\Philip\OneDrive\Lehigh\0_Proj2_UA-SA-Equifinality\ModelRunTest"}   
          
Config = {"NumSP":4,
          "PopSize": 40,            # Must be even.
          "MaxGen": 100,
          "SamplingMethod": "MC",
          "Tolerance":0.5,
          "NumEllite": 1,           # Ellite number for each SP. At least 1.
          "MutProb": 0.3,           # Mutation probability.
          "DropRecord": True,       # Population record will be dropped. However, ALL simulated results will remain. 
          "ParalCores": 1,          # This will overwrite system config.
          "AutoSave": True,         # Automatically save a model snapshot after each generation.
          "Printlevel": 10,         # Print out level. e.g. Every ten generations.
          "Plot": True              # Plot loss with Printlevel frequency.
          }
GA = DMCGA(LossFunc = HimmelblauFunc, Inputs = Inputs, Config = Config)

# %%
# Start DMCGA
GA.run()
pprint(GA.Solutions)
#GA.Best["Loss"]["SP0"]
# %%
# Continue previous run by loading into AutoSave.pickle.
GA = DMCGA(LossFunc = HimmelblauFunc, Inputs = Inputs, Config = Config, ContinueFile = r"......\AutoSave.pickle" )
GA.run()