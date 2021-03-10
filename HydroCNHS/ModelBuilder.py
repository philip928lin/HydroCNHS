r"""
ModelBuilder is a user interface for building Model.yaml.
Note that ModelBuilder only help users to establish the skeleton of Model.yaml,
which users need to manually populate values inside Model.yaml.
After populating values inside Model.yaml, ModelBuilder provide a loadmodel() staticmethod 
to check and parse Model.yaml, which ensure the eligibility of Model.yaml.
"""
from .SystemConrol import loadModel, writeModel
from copy import deepcopy
from collections import OrderedDict

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
            
class ModelBuilder(object):
    def __init__(self):
        #self.Model = OrderedDict()
        self.Model = {}
        self.Model["Path"] = {"WD": "Required"}
        self.Model["WaterSystem"] = {}
        self.Model["WaterSystem"]["StartDate"]      = "Required"
        self.Model["WaterSystem"]["NumSubbasin"]    = 0
        self.Model["WaterSystem"]["NumGauges"]      = 0
        self.Model["WaterSystem"]["NumAgents"]      = 0
        self.Model["WaterSystem"]["Outlets"]        = []
        self.Model["WaterSystem"]["GaugedOutlets"]  = []
        self.Model["WaterSystem"]["DataLength"]     = "Required"
        
    def addLSMSubbasins(self, OutletLists, model = "GWLF"):
        self.Model["WaterSystem"]["Outlets"] = OutletLists
        self.Model["WaterSystem"]["NumSubbasin"] = len(OutletLists)
        
        # Select LSM templete.
        if model == "GWLF":
            LSM_templete = GWLF
        else:
            raise ValueError("Given LSM model, {}, is not eligible.".format(model))
        
        self.Model["LSM"] = {"Model": model}
        for sub in OutletLists:
            self.Model["LSM"][sub] = deepcopy(LSM_templete)                                 
    
    def addRoutingLinks(self, GaugedOutletsDict, InstreamOutlets = [], model = "Lohmann"):
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
                if o in InstreamOutlets:
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
        if self.Model.get("ABM") is None:
            self.Model["ABM"] = {"Inputs": {"InStreamAgentTypes":  [],     
                                            "DiversionAgentTypes": []}}
        self.Model["ABM"]["Inputs"]["InStreamAgentTypes"] = list(AgentDict.keys())
        
        if Templete is None:
            Agent_templete = Agent
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
            Agent_templete = Agent
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