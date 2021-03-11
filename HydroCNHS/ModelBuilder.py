#==============================================================
# ModelBuilder.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# 2021/03/10
#==============================================================
r"""
ModelBuilder is a user interface for building Model.yaml.
Note that ModelBuilder only help users to establish the skeleton of Model.yaml,
which users need to manually populate values inside Model.yaml.
After populating values inside Model.yaml, ModelBuilder provide a loadmodel() staticmethod 
to check and parse Model.yaml, which ensure the eligibility of Model.yaml.
"""
from .SystemConrol import loadModel, writeModel
from copy import deepcopy
import os

GWLF = {"Inputs": {"Area":      "Required",
                   "Latitude":  "Required",
                   "S0":        "Required",
                   "U0":        "Required",
                   "SnowS":     "Required"},
        "Pars": {"CN2":     -99,
                 "IS":      -99,
                 "Res":     -99,
                 "Sep":     -99,
                 "Alpha":   -99,
                 "Beta":    -99,
                 "Ur":      -99,
                 "Df":      -99,
                 "Kc":      -99}}

HYMOD = {"Inputs": {"Area":      "Required",
                    "SnowS":     "Required"},
        "Pars": {"Cmax":    -99,
                 "Bexp":    -99,
                 "Alpha":   -99,
                 "Kq":      -99,
                 "Ks":      -99,
                 "Beta":    -99,
                 "Df":      -99}}

Lohmann = {"Inputs": {"FlowLength":        "Required",
                      "InStreamControl":   False},
           "Pars": {"GShape":   -99,
                    "GScale":   -99,
                    "Velo":     -99,
                    "Diff":     -99}}

Agent = {"Attributions": None,
         "Inputs": {"DMFreq":   "Required",
                    "Piority":  "Required",
                    "Links":    "Required",
                    "RL": {"ValueFunc":     "Value.Sigmoid",
                           "PolicyFunc":    "Policy.Gaussian",
                           "kwargs": {"muFunc":     "Value.Sigmoid",
                                      "sigFunc":    "Policy.Gaussian",
                                      "muIndexInfo":    None,
                                      "sigIndexInfo":   None} }},
         "Pars": {"W": "Required",
                  "Theta": "Required",
                  "LR_W": "Required",
                  "LR_T": "Required",
                  "LR_R": "Required",
                  "Lambda_W": None,
                  "Lambda_T": None}}
    
# This section is customize for YRB.
NumResModels = 5
InputDataPath = r"C:\Users\Philip\OneDrive\Lehigh\0_Proj2_UA-SA-Equifinality\YRBModel\InputData"
ResAgent = {"Attributions": {"Capacity": "Required",
                             "ObvDfPath": {"MonthlyFlow":   os.path.join(InputDataPath, "Res_MonthlyFlow(cms).csv"),
                                           "MonthlydPrep":  os.path.join(InputDataPath, "Res_MonthlydPrep(cm).csv")},
                             "InitStorage": "Required",
                             "InitResRef": "Required",
                             "Scale": {"G": 1, "C1": 1, "C2": 1, "dPTolWin": 1, 
                                       "dPR1": 1, "dPR2": 1, "dPR3": 1, 
                                       "dResSPerR1": 1, "dResSPerR2": 1, "dResSPerR3": 1, "dPResWTotal": 1,
                                       "dS1": 1, "dS2": 1, "dS3": 1}},
            "Inputs": {"DMFreq":            [-9, -9, 1],
                       "Piority":           0,
                       "ModelAssignList":   [0, 2, 1, 1, 0, 2, 4, 2, 0, 0, 3, 3], 
                       "Links":             "Required",
                       "RL":               {"ValueFunc":  "Value.Sigmoid",
                                            "PolicyFunc": "Policy.Gaussian",
                                            "kwargs":     {"muFunc":     "Policy.Linear",
                                                           "FixedSig":   True} } },
            "Pars": {"W":       [-99, -99] * NumResModels,                         
                     "Theta":   [-99, -99, -99, -99, -99] * NumResModels,
                     "LR_W":    [-99, -99] * NumResModels,
                     "LR_T":    [-99, -99, -99, -99, -99, -99] * NumResModels,
                     "LR_R":    [-99] * NumResModels,
                     "Sig":     [-99] * NumResModels}}  

DivAgent = {"Attributions": {"Area": "Required",
                             "ObvDfPath": {"AnnualFlow":    os.path.join(InputDataPath, "Div_AnnualFlow(cms).csv"),
                                           "AnnualdPrep":   os.path.join(InputDataPath, "Div_AnnualdPrep(cm).csv")},
                             "InitDivRef": "Required",
                             "Scale": {"G": 1, "C1": 1, "C2": 1, "dPTolWin": 1, 
                                       "dPR1": 1, "dPR2": 1, "dPR3": 1, 
                                       "dResSPerR1": 1, "dResSPerR2": 1, "dResSPerR3": 1, "dPResWTotal": 1,
                                       "dS1": 1, "dS2": 1, "dS3": 1}},
            "Inputs": {"DMFreq":            [-9, 3, 1],
                       "Piority":           1,
                       "Links":             "Required",
                       "RL":               {"ValueFunc":  "Value.Sigmoid",
                                            "PolicyFunc": "Policy.Gaussian",
                                            "kwargs":     {"muFunc":     "Policy.Linear",
                                                           "FixedSig":   True} } },
            "Pars": {"W":       [-99],                         
                     "Theta":   [-99, -99, -99, -99],
                     "LR_W":    [-99],
                     "LR_T":    [-99, -99, -99, -99],
                     "LR_R":    [-99],
                     "Sig":     [-99]}}  

            
class ModelBuilder(object):
    def __init__(self, WD, StartDate, DataLength):
        """Create ModelBuilder.
        Note that we temporary customize addAgent for YRB!!

        Args:
            WD (str): Working directory.
            StartDate (str): Simulation start date (e.g. YYYY/M/D).
            DataLength (str): Simulation length.
        """
        self.Model = {}
        self.Model["Path"] = {"WD": WD}
        self.Model["WaterSystem"] = {}
        self.Model["WaterSystem"]["StartDate"]      = StartDate
        self.Model["WaterSystem"]["NumSubbasin"]    = 0
        self.Model["WaterSystem"]["NumGauges"]      = 0
        self.Model["WaterSystem"]["NumAgents"]      = 0
        self.Model["WaterSystem"]["Outlets"]        = []
        self.Model["WaterSystem"]["GaugedOutlets"]  = []
        self.Model["WaterSystem"]["DataLength"]     = DataLength
        
    def addLSMSubbasins(self, OutletLists, model = "GWLF"):
        """Add sub-basin for LSM model.

        Args:
            OutletLists (list): A list of outlet names for each sub-basin.
            model (str, optional): LSM model. GWLF or HYMOD. Defaults to "GWLF".
        """
        self.Model["WaterSystem"]["Outlets"] = OutletLists
        self.Model["WaterSystem"]["NumSubbasin"] = len(OutletLists)
        
        # Select LSM templete.
        if model == "GWLF":
            LSM_templete = GWLF
        elif model == "HYMOD":
            LSM_templete = HYMOD
        else:
            raise ValueError("Given LSM model, {}, is not eligible.".format(model))
        
        self.Model["LSM"] = {"Model": model}
        for sub in OutletLists:
            self.Model["LSM"][sub] = deepcopy(LSM_templete)                                 
    
    def addRoutingLinks(self, GaugedOutletsDict, InstreamControls = [], model = "Lohmann"):
        """Add routing links for each gauged outlet.

        Args:
            GaugedOutletsDict (dict): {Gauged outlet: A list of outlets of streamflow contributors.}
            InstreamControls (list, optional): A list of instream controls' name. Defaults to [].
            model (str, optional): Routing model. Defaults to "Lohmann".
        """
        if self.Model.get("LSM") is None:
            raise ValueError("Cannot find Model['LSM']. Please run addLSMSubbasins() first.")
        
        self.Model["WaterSystem"]["GaugedOutlets"] = list(GaugedOutletsDict.keys())
        self.Model["WaterSystem"]["NumGauges"] = len(self.Model["WaterSystem"]["GaugedOutlets"])
        
        # Select LSM templete.
        if model == "Lohmann":
            Routing_templete = Lohmann
        else:
            raise ValueError("Given Routing model, {}, is not eligible.".format(model))
        
        self.Model["Routing"] = {"Model": model}
        for g, outlets in GaugedOutletsDict.items():
            self.Model["Routing"][g] = {}
            for o in outlets:
                self.Model["Routing"][g][o] = deepcopy(Routing_templete)
                if o in InstreamControls:
                    self.Model["Routing"][g][o]["Inputs"]["InStreamControl"] = True
                    self.Model["Routing"][g][o]["Pars"]["GShape"] = None
                    self.Model["Routing"][g][o]["Pars"]["GScale"] = None
            # Make sure adding g
            self.Model["Routing"][g][g] = deepcopy(Routing_templete)
            self.Model["Routing"][g][g]["Inputs"]["FlowLength"] = 0
            self.Model["Routing"][g][g]["Pars"]["Velo"] = None
            self.Model["Routing"][g][g]["Pars"]["Diff"] = None
        # Add all outlets in-grid routing.
        for o in self.Model["WaterSystem"]["Outlets"]:
            self.Model["Routing"][o] = {o: deepcopy(Routing_templete)}
            self.Model["Routing"][o][o]["Inputs"]["FlowLength"] = 0
            self.Model["Routing"][o][o]["Pars"]["Velo"] = None
            self.Model["Routing"][o][o]["Pars"]["Diff"] = None
        
    def addInStreamAgents(self, AgentDict, Templete = None):
        """Add instream agents (controls).

        Args:
            AgentDict (dict): {AgType: A list of agent name.}
            Templete (dict, optional): Templete dictionary. Defaults to None.
        """
        if self.Model.get("ABM") is None:
            self.Model["ABM"] = {"Inputs": {"InStreamAgentTypes":  [],     
                                            "DiversionAgentTypes": []}}
        self.Model["ABM"]["Inputs"]["InStreamAgentTypes"] = list(AgentDict.keys())
        
        if Templete is None:
            Agent_templete = ResAgent   # Customize for YRB.
        else:
            Agent_templete = Templete
        
        for agType, agList in AgentDict.items():
            self.Model["ABM"][agType] = {}
            for ag in agList:
                self.Model["ABM"][agType][ag] = deepcopy(Agent_templete)

        # Update WaterSyetem 
        count = 0
        for agType, agList in self.Model["ABM"].items():
            if agType != "Inputs":
                count += len(agList)
        self.Model["WaterSystem"]["NumAgents"] = count
    
    def addInDiversionAgents(self, AgentDict, Templete = None):
        if self.Model.get("ABM") is None:
            self.Model["ABM"] = {"Inputs": {"InStreamAgentTypes":  [],     
                                            "DiversionAgentTypes": []}}
        self.Model["ABM"]["Inputs"]["DiversionAgentTypes"] = list(AgentDict.keys())
        
        if Templete is None:
            Agent_templete = DivAgent
        else:
            Agent_templete = Templete
        
        for agType, agList in AgentDict.items():
            self.Model["ABM"][agType] = {}
            for ag in agList:
                self.Model["ABM"][agType][ag] = deepcopy(Agent_templete)

        # Update WaterSyetem 
        count = 0
        for agType, agList in self.Model["ABM"].items():
            if agType != "Inputs":
                count += len(agList)
        self.Model["WaterSystem"]["NumAgents"] = count    
    
    def getModelDict(self):
        return self.Model
    
    def to_yaml(self, Filename):
        writeModel(self.Model, Filename)
        
    
    @staticmethod
    def verifyModel(Model):
        loadModel(Model)