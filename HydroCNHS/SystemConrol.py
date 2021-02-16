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

    
def loadModel(model, Checked = False, Parsed = False):
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
        
    # Check model is consist and correct.
    if Checked is not True:     # We don't recheck model is, it has been checked.
        if checkModel(Model) is not True:
            return None
    
    # Parse model
    if Parsed is not True:
        Model = parseModel(Model)
    
    return Model

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
    Model["SystemParsedData"] = {}
    
    if Model.get("Routing") is not None :
        Model = parseSimulationSeqence(Model)
    return Model    
    

def checkInStreamAgentInRouting(Model):
    """To make sure InStreamAgentInflows outlets are assigned in the routing section.
    """
    # Untest yet
    Routing = Model["Routing"]
    
    # Add InStreamAgents for routing model check if ABM and Routing exists.
    InStreamAgentInflows = []        # In-stream agents
    InStreamAgentTypes = Model["ABM"]["Inputs"]["InStreamAgentTypes"]
    for ISagType in InStreamAgentTypes:
        for ag in Model["ABM"][ISagType]:
            Links = Model["ABM"][ISagType][ag]["Inputs"]["Links"]
            InStreamAgentInflows += [outlet for outlet in Links if Links[outlet] == -1]
    InStreamAgentInflows = list(set(InStreamAgentInflows))        # Eliminate duplicates.
    #Model["SystemParsedData"]["InStreamAgentInflows"] = InStreamAgentInflows   # Add to system parsed data.

    # Check InStreamAgents' inflow outlets are in RoutingOutlets
    RoutingOutlets = list(Routing.keys())
    RoutingOutlets.remove('Model')  
    for vro in InStreamAgentInflows:
        if vro not in RoutingOutlets:
            logger.error("[Check model failed] Cannot find in-stream agent's inflow outlets in the Routing section. Routing outlets should include {}.".format(vro))
            return False
    else:
        for end in InStreamAgentInflows:
            for start in Routing[end]:
                # Check if start belong to others RoutingOutlets' starts.
                for ro in RoutingOutlets:
                    if ro != end:
                        if any( start in others for others in Routing[ro] ):
                            if ro in Routing[end]:  # If it is in the upstream of VirROutlet, it is fine.
                                pass
                            else:
                                logging.error("[Check model failed] {} sub-basin outlet shouldn't be in {}'s (routing outlet) catchment outlets due to the seperation of in-stream control agents.".format(start, end, ro))
                                return False
        return True

def parseSimulationSeqence(Model):
    SystemParsedData = Model["SystemParsedData"]
    SystemParsedData["SimSeq"] = None
    SystemParsedData["AgSimSeq"] = None
    SystemParsedData["RoutingOutlets"] = None
    SystemParsedData["InStreamAgents"] = None
    
    #----- Collect in-stream agents
    InStreamAgents = []
    if Model.get("ABM") is not None:
        ABM = Model["ABM"]
        for agType in ABM["Inputs"]["InStreamAgentTypes"]:
            for end in ABM[agType]:
                InStreamAgents.append(end)
        Model["SystemParsedData"]["InStreamAgents"] = InStreamAgents
        
    #----- Step1: Form the simulation sequence of routing outlets and in-stream control agents -----
    # Collected edges and track-back dictionary. 
    Edges = []                              # This can be further used for plotting routing simulation schema using networkx.
    BackTrackingDict = {}
    Routing = Model["Routing"]
    RoutingOutlets = list(Routing.keys())
    RoutingOutlets.remove('Model')  
    for end in RoutingOutlets:
        for start in Routing[end]:
            if start == end:
                pass   # We don't add self edge. 
            else:
                if start in RoutingOutlets+InStreamAgents: # Eliminate only-sub-basin outlets. If need full stream node sequence, remove this.
                    Edges.append((start, end))
                    if BackTrackingDict.get(end) is None:
                        BackTrackingDict[end] = [start]
                    else:
                        BackTrackingDict[end].append(start)
    
    # Add in-stream agents connectinons if ABM sections exists.     
    if Model.get("ABM") is not None:
        ABM = Model["ABM"]
        for agType in ABM["Inputs"]["InStreamAgentTypes"]:
            for end in ABM[agType]:
                Links = ABM[agType][end]["Inputs"]["Links"]
                InflowNodes = [ node for node in Links if Links[node] == -1] 
                for start in InflowNodes:
                    Edges.append((start, end))
                    if BackTrackingDict.get(end) is None:
                        BackTrackingDict[end] = [start]
                    else:
                        BackTrackingDict[end].append(start)
    
    # Back tracking to form simulation sequence.        
    def formSimSeq(Node, BackTrackingDict, GaugedOutlets):
        """A recursive function, which keep tracking back upstream nodes until reach the most upstream one. We design this in a clear way. To understand to logic behind, please run step-by-step using the test example at the bottom of the code.
        """
        SimSeq = []
        def trackBack(node, SimSeq, BackTrackingDict, TempDict = {}, AddNode = True):
            if AddNode:
                SimSeq = [node] + SimSeq
            if BackTrackingDict.get(node) is not None:
                gaugedOutlets = [o for o in BackTrackingDict[node] if o in GaugedOutlets] 
                if len(gaugedOutlets) >= 1:
                    # g > 1 or len(g) < len(all) => update TempDict
                    if len(gaugedOutlets) > 1 or len(gaugedOutlets) < len(BackTrackingDict[node]): 
                        # Get rank of each g
                        rank = []
                        for g in gaugedOutlets:
                            upList = BackTrackingDict.get(g)
                            if upList is None:
                                rank.append(0)
                            else:
                                rank.append(len(upList))
                        gmax = gaugedOutlets[rank.index(max(rank))]
                        # Update TempDict: delete node and update others.
                        gaugedOutlets.remove(gmax)
                        TempDict.pop(gmax, None)    # if 'key' in my_dict: del my_dict['key']
                        for g in gaugedOutlets:
                            TempDict[g] = node
                        # Call trackBack with gmax and TempDict (recursive)
                        SimSeq, TempDict, BackTrackingDict = trackBack(gmax, SimSeq, BackTrackingDict, TempDict)
                        
                    elif len(gaugedOutlets) == 1 and len(BackTrackingDict[node]) == 1:
                        SimSeq, TempDict, BackTrackingDict = trackBack(gaugedOutlets[0], SimSeq, BackTrackingDict, TempDict)
                        TempDict.pop(gaugedOutlets[0], None)
                        # Search TempDict and jump backward to add other tributary.
                        # reverse TempDict
                        rTempDict = {}
                        for g in TempDict:
                            if rTempDict.get(TempDict[g]) is None:
                                rTempDict[TempDict[g]] = [g]
                            else:
                                rTempDict[TempDict[g]].append(g)
                        if rTempDict != {}:
                            # Replace BackTrackingDict
                            for g in rTempDict:
                                BackTrackingDict[g] = rTempDict[g]
                            ToNode = SimSeq[min([SimSeq.index(i) for i in rTempDict])]
                            SimSeq, TempDict, BackTrackingDict = trackBack(ToNode, SimSeq, BackTrackingDict, {}, False)
                else:
                    for up in BackTrackingDict[node]:
                        SimSeq, TempDict, BackTrackingDict = trackBack(up, SimSeq, BackTrackingDict, TempDict)    
            return SimSeq, TempDict, BackTrackingDict 
        SimSeq, TempDict, BackTrackingDict = trackBack(Node, SimSeq, BackTrackingDict)
        return SimSeq
    
    
    # LastNode = End nodes - start node. Last node is the only one that does not exist in start nodes.
    LastNode = list(set(BackTrackingDict.keys()) - set([i[0] for i in Edges]))[0]   
    GaugedOutlets = Model["WaterSystem"]["GaugedOutlets"]
    SimSeq = formSimSeq(LastNode, BackTrackingDict, GaugedOutlets)
    SystemParsedData["SimSeq"] = SimSeq
    # Sort RoutingOutlets to SimSeq
    SystemParsedData["RoutingOutlets"] = [ro for ro in SimSeq if ro in RoutingOutlets] 
    #-----------------------------------------------------------------------------------------------
    
    #----- Step2: Form AgSim dictionary -----
    # Aggregate to only SimSeq's node. 
    if Model.get("ABM") is not None:
        InStreamAgentTypes = Model["ABM"]["Inputs"]["InStreamAgentTypes"]
        DiversionAgentTypes = Model["ABM"]["Inputs"]["DiversionAgentTypes"]
        RoutingOutlets = SystemParsedData["RoutingOutlets"]     # Ordered
        
        def searchRoutingOutlet(agQ):
            """Find in which routing outlet first need agQ to adjust original Q. (from upstream).
            Args:
                agQ (str): Outlets connection of an agent.
            Returns:
                str: Routing outlet.
            """
            if agQ in RoutingOutlets:
                return agQ
            for ro in RoutingOutlets:
                if agQ in Routing[ro]:
                    return ro

        AgSimSeq = {}              ; Piority = {}
        AgSimSeq["AgSimMinus"] = {}; Piority["AgSimMinus"] = {}
        AgSimSeq["AgSimPlus"] = {} ; Piority["AgSimPlus"] = {}
        for ss in SimSeq:
            AgSimSeq["AgSimMinus"][ss] = []; Piority["AgSimMinus"][ss] = []
            AgSimSeq["AgSimPlus"][ss] = [] ; Piority["AgSimPlus"][ss] = []
        
        for agType in InStreamAgentTypes:
            for ag in Model["ABM"][agType]:
                # No AgSimMinus is needed since in-stream agents replace original streamflow.
                AgSimSeq["AgSimPlus"][ag].append(ag)
                Piority["AgSimPlus"][ag].append(0)  # In-stream agent always has piority 0.
                
        for agType in DiversionAgentTypes:
            for ag in Model["ABM"][agType]:
                Links = Model["ABM"][agType][ag]["Inputs"]["Links"]
                Plus = [p for p in Links if Links[p] >= 0]
                Minus = [m for m in Links if Links[m] <= 0]
                for p in Plus:
                    ro = searchRoutingOutlet(p)
                    AgSimSeq["AgSimPlus"][ro].append(ag)
                    Piority["AgSimPlus"][ro].append(Model["ABM"][agType][ag]["Inputs"]["Piority"])
                for m in Minus:
                    ro = searchRoutingOutlet(m)
                    AgSimSeq["AgSimMinus"][ro].append(ag)
                    Piority["AgSimMinus"][ro].append(Model["ABM"][agType][ag]["Inputs"]["Piority"])

        # Sort agents based on their piorities               
        for pm in AgSimSeq:
            for ro in AgSimSeq[pm]:
                Agents = AgSimSeq[pm][ro]
                Piorities = Piority[pm][ro]
                Agents = [ag for _,ag in sorted(zip(Piorities,Agents))]
                AgSimSeq[pm][ro] = list(set(Agents))    # Remove duplicated ags.

        SystemParsedData["AgSimSeq"] = AgSimSeq    
        #----------------------------------------
    Model["SystemParsedData"] = SystemParsedData
    
    return Model
#-------------------------------------

#%% Test
r"""
BackTrackingDict = {"G":["g7","g8","R1"],
                    "g7":["R1"],
                    "R1":["V1"],
                    "V1":["g1","g2","g3","g4","g5","g6"],
                    "g6":["g1","g2","g3","g4","g5"],
                    "g3":["g1","g2"],
                    "g2":["g1"],
                    "g5":["g4"],
                    "g8":["g7","R1"]}
GaugedOutlets = ["g1","g2","g3","g4","g5","g6","g7","g8","G"]
Node = "G"


SimSeq = formSimSeq(Node, BackTrackingDict, GaugedOutlets)
# Expect to get ['g4', 'g5', 'g1', 'g2', 'g3', 'g6', 'V1', 'R1', 'g7', 'g8', 'G']

"""