#%%
# Form the water system using a semi distribution hydrological model and agent-based model.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# 2021/02/05

from .HydroProcess import runGWLF, calPEt_Hamon
from .RiverRouting import formUH_Lohmann, runTimeStep_Lohmann
from .SystemConrol import loadConfig, loadModel
import logging
logger = logging.getLogger("HydroCNHS") # Get logger (console only)

class HydroCNHS(object):
    """Main HydroCNHS simulation object.

    """
    def __init__(self, model):
        # Load HydroCNHS system configuration.
        self.config = loadConfig()      
        if self.config["OutputLogFile"]:
            logger = logging.getLogger("runHydroCNHS") # Get logger that will output to both console and log file.
        
        # Load model and distribute into several variables.
        Model = loadModel(model)        # User-provided model
        self.Path = Model["Path"]
        self.WS = Model["WaterSystem"]     # WS: Water system
        self.HP = Model["HydroProcess"]     # HP: Hydrological process
        self.RR = Model["RiverRouting"]     # RR: River routing 
        self.ABM = Model["ABM"] 

        # initialize output
        self.Q = {}     # [cms] Streamflow for each outlet.
        self.A = {}     # Collect output for AgentType Agent its outputs
    
    def loadWeatherData(self, T, P):
        """Load temperature and precipitation data.
        CAn add some check function or deal with miss value here.
        Args:
            T (dict): [degC] Daily mean temperature time series data (value) for each sub-basin named by its outlet.
            P (dict): [cm] Daily precipitation time series data (value) for each sub-basin named by its outlet.
        """
        self.Obv = {"T":T, "P":P}
        logger.info("Load T & P with total length {}.".format(self.WS["DataLength"]))
        
    def run(self):
        # Set a timer here
        
        # GWLF
        calPEt_Hamon(Tt, Lat, StartDate, dz = None)
        runGWLF(GWLFPars, Inputs, Tt, Pt, PEt, StartDate, DataLength)
        
        # Form UH
        UH_Lohmann = {}
        formUH_Lohmann(FlowLen, RoutePars)
        
        # Time step simulation (Coupling hydrological model and ABM)
        for t in (self.WS["DataLength"]):
            
            # Update Qt ABM
            if .....
            
            runTimeStep_Lohmann(self.WS["GaugedOutlets"], self.RR, UH_Lohmann, self.Q, t)
        
        logger.info("Complete HydroCNHS simulation".format(self.WS["DataLength"]))
        return None
        
        
        
# #---- Loading input ----------
# # Load Config
# # Load model
# with open('Config.yaml', 'rt') as file:
#         config = yaml.safe_load(file.read())
#         WD = config["Path"]["WD"]
#         print(config)




# 	directflow = np.zeros(len(Qgwlf))
# 	for i in range (0, len(Qgwlf)):
# 		for j in range (0, KE+UH_DAY-1):
# 			if (i-j+1) >= 1:
# 				directflow[i]= directflow[i] + UH_direct[j] * Qgwlf[i-j]
    
# import yaml
# from pprint import pprint
# def loadModel(model):
#     """Load model and conduct initial check for the consistency.

#     Args:
#         model (str): Model filename. Has to be .yaml file.
#     """
#     with open(model, 'rt') as file:
#         Model = yaml.safe_load(file.read()) 
#     return Model

# model = loadModel(r"C:\Users\Philip\OneDrive\Lehigh\0_Proj2_UA-SA-Equifinality\ModelRunTest\model.yaml")
# RiverRouting = model["RiverRouting"]
# pprint(RiverRouting)