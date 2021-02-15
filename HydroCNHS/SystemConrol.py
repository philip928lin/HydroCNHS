#%%
# System control file for HydroCNHS.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# 2021/02/05

import logging
import logging.config
import traceback
from joblib import logger
import yaml
import ruamel.yaml      # For round trip modification (keep comments)
import os 
logger = logging.getLogger("HydroCNHS.SC") # Get logger 

r"""
We need to modify yaml, which we can load and write the file while keeping comments.
https://stackoverflow.com/questions/7255885/save-dump-a-yaml-file-with-comments-in-pyyaml/27103244
"""
#-----------------------------------------------
#---------- Read and Wright Functions ----------

this_dir, this_filename = os.path.split(__file__)
def loadConfig():
    """Get config dictionary from Config.yaml.

    Returns:
        dict: Dictionary of model config.
    """
    #print(os.path.join(this_dir, 'Config.yaml'))
    with open(os.path.join(this_dir, 'Config.yaml'), 'rt') as file:
        config = yaml.safe_load(file.read())
    return config

def updateConfig(ModifiedConfig):
    """Given the dictionary of modified setting, this funciton will over write Config.yaml.

    Args:
        Config (dict): Dictionary of modified config setting
    """
    yaml_round = ruamel.yaml.YAML()  # defaults to round-trip if no parameters given
    with open(os.path.join(this_dir, 'Config.yaml'), 'rt') as file:
        config = yaml_round.load(file.read())
        
    # Relace values
    for key in ModifiedConfig:
        if isinstance(ModifiedConfig, dict):    # Second level
            for key2 in ModifiedConfig[key]:
                config[key][key2] = ModifiedConfig[key][key2]
        else:                                   # First level
            config[key] = ModifiedConfig[key]
            
    with open('Config.yaml', 'w') as file:
        yaml_round.dump(config, file)


def defaultConfig():
    """Repalce Config.yaml back to default setting.
    """
    yaml_round = ruamel.yaml.YAML()  # defaults to round-trip if no parameters given
    with open('Config_default.yaml', 'rt') as file:
        Config_default = yaml_round.load(file.read())
    with open('Config.yaml', 'w') as file:
        yaml_round.dump(Config_default, file)

    
def loadModel(model, Checked = False):
    """Load model and conduct initial check for its setting consistency.

    Args:
        model (str): Model filename. Has to be .yaml file.
    """
    logger = logging.getLogger("HydroCNHS")
    
    if isinstance(model, dict):     # For expert of HydroCNHS to use. Give model in a dictionary form directly.
        Model = model
    else:
        with open(model, 'rt') as file: 
            try:
                Model = yaml.safe_load(file.read())
            except Exception as e:
                logger.error(traceback.format_exc())   # Logs the error appropriately.
                return None
    
    
    # Create SystemParsedData dictionary. This will be used to update model.yaml if successfully parse and check the model.
    SystemParsedData = {}

    # Add VirROutlets for routing model check if ABM and Routing exists.
    VirROutlets = []        # In-stream agents
    if Model.get("Routing") is not None and Model.get("ABM") is not None:
        InStreamAgentTypes = Model["ABM"]["Inputs"]["InStreamAgentTypes"]
        for ISagType in InStreamAgentTypes:
            for ag in Model["ABM"][ISagType]:
                Links = Model["ABM"][ISagType][ag]["Inputs"]["Links"]
                VirROutlets += [outlet for outlet in Links if Links[outlet] == -1]
        VirROutlets = list(set(VirROutlets))        # Eliminate duplicates.
    SystemParsedData["VirROutlets"] = VirROutlets   # Add to system parsed data
        
    # Check model is consist and correct.
    if Checked is not True:     # We don't recheck model is, it has been checked.
        if checkModel(Model) is not True:
            return None
    
    # Parse model
    parseModel(Model)

#-----------------------------------------------


#-----------------------------------------
#---------- Auxiliary Functions ----------

def Dict2String(Dict, Indentor = "  "):
    def Dict2StringList(Dict, Indentor = "  ", count = 0, string = []):
        for key, value in Dict.items(): 
            string.append(Indentor * count + str(key))
            if isinstance(value, dict):
                string = Dict2StringList(value, Indentor, count+1, string)
            else:
                string.append(Indentor * (count+1) + str(value))
        return string
    return "\n".join(Dict2StringList(Dict, Indentor))
#-----------------------------------------



#-------------------------------------
#---------- Check and Parse Functions ----------
def checkModel(Model):
    """Check the consistency of the model dictionary.

    Args:
        model (dict): Loaded from model.yaml. 

    Returns:
        bool: True if pass the check.
    """
    Pass = True
    if Model.get("Routing") is not None and Model.get("ABM") is None:
        pass
    
    if Model.get("Routing") is not None and Model.get("ABM") is not None:
        Pass = checkInStreamAgentInRouting(Model)
    return Pass

def parseModel(Model):
    """Parse model dictionary. Populate SystemParsedData.

    Args:
        Model (dict): Load from model.yaml.

    Returns:
        dict: Model
    """
    Model = parseSimulationSeqence(Model)
    return Model    
    

def checkInStreamAgentInRouting(Model):
    # Untest yet
    Routing = Model["Routing"]
    VirROutlets = Model["SystemParsedData"]["VirROutlets"]

    # Check VirROutlets are in RoutingOutlets
    RoutingOutlets = list(Routing.keys())
    RoutingOutlets.remove('Model')  
    if any( vro not in RoutingOutlets for vro in VirROutlets ):
        logger.error("[Load model failed] Cannot find in-stream control objects inflow outlets in Routing section. Routing outlets should include {}.".format(VirROutlets))
        return False
    else:
        for end in VirROutlets:
            for start in Routing[end]:
                # Check if start belong to others RoutingOutlets' starts.
                for ro in RoutingOutlets:
                    if ro != end:
                        if any( start in others for others in Routing[ro] ):
                            if ro in Routing[end]:  # If it is in the upstream of VirROutlet, it is fine.
                                pass
                            else:
                                logging.error("[Load model failed] {} in {}'s catchment outlets shouldn't belong to routing outlet {}'s catchment outlets. (Seperated by in-stream objects)".format(start, end, ro))
                                return False
        return True

def parseSimulationSeqence(Model):
    SystemParsedData = Model["SystemParsedData"]
    SystemParsedData["SimSeq"] = None
    SystemParsedData["AgSimSeq"] = None
    
    if Model.get("Routing") is not None :
        #----- Step1: Form the simulation sequence of routing outlets and in-stream control agents -----
        # Collected edges and track-back dictionary. 
        Edges = []                              # This can be further used for plotting simulation schema using networkx.
        BackTrackingDict = {}
        Routing = Model["Routing"]
        RoutingOutlets = list(Routing.keys())
        RoutingOutlets.remove('Model')  
        for end in RoutingOutlets:
            for start in Routing["end"]:
                Edges.append((start, end))
                if BackTrackingDict.get(end) is None:
                    BackTrackingDict[end] = [start]
                else:
                    BackTrackingDict[end].append(start)
        
        # Add in-stream agents connectinons if ABM sections exists.             
        if Model.get("ABM") is not None:
            ABM = Model["ABM"]
            for agType in ABM["InStreamAgentTypes"]:
                for end in ABM[agType]:
                    Links = ABM[agType][end]["Inputs"]["Links"]
                    InflowNodes = [ node for node in Links if Links[node] == -1]
                    for start in InflowNodes:
                        Edges.append((start, end))
                        BackTrackingDict[end].append(start)     # InflowNodes should already be in keys of BackTrackingDict.
        
        # Back tracking to form simulation sequence.        
        def formSimSeq(Node, BackTrackingDict):
            """A recursive function, which keep tracking back upstream nodes until reach the most upstream one.
            """
            SimSeq = []
            def trackBack(node, SimSeq, BackTrackingDict):
                SimSeq = [node] + SimSeq
                if BackTrackingDict.get(node) is not None:
                    for up in BackTrackingDict[node]:
                        SimSeq = trackBack(up, SimSeq, BackTrackingDict)    
                return SimSeq 
            return trackBack(Node, SimSeq, BackTrackingDict)
        # LastNode = End nodes - start node. Last node is the only one that does not exist in start nodes.
        LastNode = list(set(BackTrackingDict.keys()) - set([i[0] for i in Edges]))[0]   
        SimSeq = formSimSeq(LastNode, BackTrackingDict)
        SystemParsedData["SimSeq"] = SimSeq
        #-----------------------------------------------------------------------------------------------
        
        #----- Step2: Form AgSim dictionary -----
        if Model.get("ABM") is not None:
            InStreamAgentTypes = Model["ABM"]["Inputs"]["InStreamAgentTypes"]
            DiversionAgentTypes = Model["ABM"]["Inputs"]["DiversionAgentTypes"]
            
            AgSimSeq = {}
            AgSimSeq["AgSimMinus"] = {}
            AgSimSeq["AgSimPlus"] = {}
            Piority = AgSimSeq.copy()   # To store agent's piority
            for agType in InStreamAgentTypes:
                for ag in Model["ABM"][agType]:
                    # No AgSimMinus is needed since in-stream agents replace original streamflow.
                    AgSimSeq["AgSimPlus"][ag] = [ag]
                    Piority["AgSimPlus"][ag] = [0]  # In-stream agent always has piority 1.
            for agType in DiversionAgentTypes:
                for ag in Model["ABM"][agType]:
                    Links = Model["ABM"][agType]
                    Plus = [p for p in Links if Links[p] >= 0]
                    Minus = [m for m in Links if Links[m] <= 0]
                    for p in Plus:
                        if AgSimSeq["AgSimPlus"].get(p) is None:      
                            AgSimSeq["AgSimPlus"][p] = [ag]
                            Piority["AgSimPlus"][p] = [Model["ABM"][agType][ag]["Inputs"]["Piority"]]
                        else:
                            AgSimSeq["AgSimPlus"][p].append(ag)
                            Piority["AgSimPlus"][p].append(Model["ABM"][agType][ag]["Inputs"]["Piority"])
                    for m in Minus:
                        if AgSimSeq["AgSimMinus"].get(m) is None:      
                            AgSimSeq["AgSimMinus"][m] = [ag]
                            Piority["AgSimMinus"][m] = [Model["ABM"][agType][ag]["Inputs"]["Piority"]]
                        else:
                            AgSimSeq["AgSimMinus"][m].append(ag)
                            Piority["AgSimMinus"][m].append(Model["ABM"][agType][ag]["Inputs"]["Piority"])
            # Sort agents based on their piorities               
            for pm in AgSimSeq:
                for ro in AgSimSeq[pm]:
                    Agents = AgSimSeq[pm][ro]
                    Piorities = Piority[pm][ro]
                    Agents = [ag for _,ag in sorted(zip(Piorities,Agents))]
                    AgSimSeq[pm][ro] = Agents  
                    
            # Aggregate to routing outlets.   
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            SystemParsedData["AgSimSeq"] = AgSimSeq    
            #----------------------------------------
    Model["SystemParsedData"] = SystemParsedData
    
    return Model
#-------------------------------------

#%% Test
r"""
from pprint import pprint
pprint(loadConfig())
initialize(WD = r"C:\Users\Philip\OneDrive\Lehigh\0_Proj2_UA-SA-Equifinality\ModelRunTest")


BackTrackingDict = {"G":["R3"],
                            "F":["R2"],
                            "D":["R1"],
                            "E":["B"],
                            "B":["A"],
                            "R1":["C"],
                            "R2":["D","E"],
                            "R3":["F"]}
a = formSimSeq("G", BackTrackingDict)
"""