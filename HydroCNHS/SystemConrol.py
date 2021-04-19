#%%
# System control file for HydroCNHS.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# 2021/02/05

import logging
import logging.config
import traceback
import pandas as pd
import yaml
import ruamel.yaml      # For round trip modification (keep comments)
import os 
import ast
import numpy as np
from copy import deepcopy   # For deepcopy dictionary.
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
        
    # Replace values
    for key in ModifiedConfig:
        if isinstance(ModifiedConfig, dict):    # Second level
            for key2 in ModifiedConfig[key]:
                config[key][key2] = ModifiedConfig[key][key2]
        else:                                   # First level
            config[key] = ModifiedConfig[key]
            
    with open(os.path.join(this_dir, 'Config.yaml'), 'w') as file:
        yaml_round.dump(config, file)

def defaultConfig():
    """Repalce Config.yaml back to default setting.
    """
    yaml_round = ruamel.yaml.YAML()  # defaults to round-trip if no parameters given
    with open(os.path.join(this_dir, 'Config_default.yaml'), 'rt') as file:
        Config_default = yaml_round.load(file.read())
    with open(os.path.join(this_dir, 'Config.yaml'), 'w') as file:
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

def writeModel(modelDict, modelname, org_model = None):
    """Output model to yaml file. If org_model is given, comments in the original file will be kept in the output model file.

    Args:
        modelDict (dict): HydroCNHS model dictionary.
        modelname (str): Output model name (e.g. ...yaml).
        org_model (str, optional): Original model name. Defaults to None.
    """
    if org_model is not None:   # Contain comments in the original model file.
        yaml_round = ruamel.yaml.YAML()  # defaults to round-trip if no parameters given
        with open(os.path.join(org_model), 'rt') as file:
            model = yaml_round.load(file.read())
            model = modelDict
        with open(os.path.join(modelname), 'w') as file:
            yaml_round.dump(model, file)
    else:                       # Dump without comments in the original model file.
        with open(modelname, 'w') as file:
            SavedModel = yaml.safe_dump(modelDict, file, sort_keys=False, default_flow_style=None)
    logger.info("Model is saved at {}.".format(modelname))

def writeModelToDF(modelDict, KeyOption = ["Pars"], Prefix = ""):
    """Write model (dictionary) to seperated dataframe for each section.

    Args:
        modelDict (dict): HydroCNHS model.
        KeyOption (list, optional): Output items: Pars, Inputs, Attributions. Defaults to ["Pars"].
        Prefix (str, optional): Prefix for the file name. Defaults to "".
    Return:
        OutputDFList, DFName
    """
    def convertDictToDF(Dict, colname):
        df = {}
        for k in Dict:
            if isinstance(Dict[k], (int, float, type(None))):
                df[k] = Dict[k]
            elif isinstance(Dict[k], list):
                for i, v in enumerate(Dict[k]):
                    df[k+".{}".format(i)] = v
            else:
                df[k] = str(Dict[k])
        df = pd.DataFrame.from_dict(df, orient="index", columns=[colname])
        return df
    
    def mergeDicts(DictList):
        DictList = list(filter(None, DictList))     # Remove None
        Len = len(DictList)
        if Len == 0:
            return {}
        ResDict = deepcopy(DictList[0])             # !! So we won't modify original dict. 
        if Len > 1:
            for d in range(1,Len):
                ResDict.update(DictList[d])
        return ResDict
    
    AllowedOutputSections = ["LSM", "Routing", "ABM"]
    SectionList = [i for i in AllowedOutputSections if i in modelDict]
    DFName = []; OutputDFList = [] 
    for s in SectionList:
        if s == "LSM" and KeyOption != ["Attributions"]:
            DFName.append(Prefix + "LSM_" + modelDict[s]["Model"])
            df = pd.DataFrame()
            for sub in modelDict[s]:
                if sub != "Model":
                    DictList = [modelDict[s][sub].get(i) for i in KeyOption]
                    MergedDict = mergeDicts(DictList)
                    convertedDict = convertDictToDF(MergedDict, sub)
                    df = pd.concat([df, convertedDict], axis=1)
            OutputDFList.append(df)   
        elif s == "Routing" and KeyOption != ["Attributions"]:
            DFName.append(Prefix + "Routing_" + modelDict[s]["Model"])
            df = pd.DataFrame()
            for ro in modelDict[s]:
                if ro != "Model":
                    for o in modelDict[s][ro]:
                        DictList = [modelDict[s][ro][o].get(i) for i in KeyOption]
                        MergedDict = mergeDicts(DictList)
                        convertedDict = convertDictToDF(MergedDict, (o, ro))
                        df = pd.concat([df, convertedDict], axis=1)
            OutputDFList.append(df)   
        elif s == "ABM":
            DFName.append(Prefix + "ABM")
            df = pd.DataFrame()
            AgTypes = modelDict[s]["Inputs"]["ResDamAgentTypes"]+ \
                      modelDict[s]["Inputs"]["RiverDivAgentTypes"]+ \
                      modelDict[s]["Inputs"]["DamDivAgentTypes"]+ \
                      modelDict[s]["Inputs"]["InsituDivAgentTypes"]
            for agtype in AgTypes:
                for ag in modelDict[s][agtype]:
                    DictList = [modelDict[s][agtype][ag].get(i) for i in KeyOption]
                    MergedDict = mergeDicts(DictList)
                    convertedDict = convertDictToDF(MergedDict, (ag, agtype))
                    df = pd.concat([df, convertedDict], axis=1)
            OutputDFList.append(df) 
    return OutputDFList, DFName

def writeModelToCSV(FolderPath, modelDict, KeyOption = ["Pars"], Prefix = ""):
    """Write model (dictionary) to seperated csv files for each section.

    Args:
        FolderPath (str): Folder path for output files.
        modelDict (dict): HydroCNHS model.
        KeyOption (list, optional): Output items: Pars, Inputs, Attributions. Defaults to ["Pars"].
        Prefix (str, optional): Prefix for the file name. Defaults to "".
    """
    OutputDFList, DFName = writeModelToDF(modelDict, KeyOption, Prefix)
    DFName = [i+".csv" for i in DFName]
    for i, df in enumerate(OutputDFList):
        df.to_csv(os.path.join(FolderPath, DFName[i]))
    logger.info("Output files {} at {}.".format(DFName, FolderPath))
    return DFName       # CSV filenames

def loadDFToModelDict(modelDict, DF, Section, Key):
    """Load dataframe to model dictionary. The dataframe has to be in certain format that generate by ModelBuilder.

    Args:
        modelDict (dict): Model dictionary.
        DF (DataFrame): DataFrame with certain format.
        Section (str): LSM or Routing or ABM.
        Key (str): Inputs or Pars or Attributions.
    
    Return:
        (dict) updated modelDict.
    """

    def parseDFToDict(df):
        def parse(i):
            try:
                val = ast.literal_eval(i)
                return val
            except:
                return i    
        
        def toNativePyType(val):
            # Make sure the value is Native Python Type, which can be safely dump to yaml.
            if "numpy" in str(type(val)):
                if np.isnan(val):   # Convert df's null value, which is np.nan into None.
                    return None     # So yaml will displat null instead of .nan
                else:
                    val = val.item()
                    if val == -99: val = int(val)       # HydroCNHS special setting.
                    return val
            else:   # Sometime np.nan is not consider as numpy.float type.
                if np.isnan(val):   # Convert df's null value, which is np.nan into None.
                    return None     # So yaml will displat null instead of .nan
                else:
                    return val      # return other type
        
        # Form a corresponding dictionary for Pars.
        Col = [parse(i) for i in df.columns]        # Since we might have tuples.
        df.columns = Col
        # Identify list items by ".".
        Ind = [i.split(".")[0] for i in df.index]   
        Ind_dup = {v: (Ind.count(v) if "." in df.index[i] else 0) for i, v in enumerate(Ind)}
        # Parse entire df
        df = df.applymap(parse)
        # Form structured dictionary with only native python type object.
        Dict = {}
        for i in Col:   # Add each sub model setting to the dict. The value is either list or str or float or int.
            temp = {}
            for par in Ind_dup:
                if Ind_dup[par] > 0:
                    temp[par] = list(df.loc[[par+"."+str(k) for k in range(Ind_dup[par])] ,[i]].to_numpy().flatten())
                    temp[par] = [toNativePyType(val) for val in temp[par]]
                    temp[par] = [val for val in temp[par] if val is not None]   # Ensure W, Theta, LR ... lists won't contain None. 
                else:
                    temp[par] = toNativePyType(df.loc[par, [i]].values[0])
            Dict[i] = temp
        return Dict
    
    #modelDict = deepcopy(modelDict)     # So we don't modify original Model dict.
    DFDict = parseDFToDict(DF)          # DF to Dict
    
    #Replace the original modelDict accordingly.
    if Section == "LSM":
        for sub in DFDict:
            modelDict["LSM"][sub][Key] = DFDict[sub]
    elif Section == "Routing":
        for roo in DFDict:
            modelDict["Routing"][roo[1]][roo[0]][Key] = DFDict[roo]
    elif Section == "ABM":
        for agagType in DFDict:
            modelDict["ABM"][agagType[1]][agagType[0]][Key] = DFDict[agagType]
                    
    return modelDict



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

def setSeed(seed):
    np.random.seed(seed)
#-----------------------------------------



#-----------------------------------------------
#---------- Check and Parse Functions ----------
def checkModel(Model):
    """Check the consistency of the model dictionary.

    Args:
        model (dict): Loaded from model.yaml. 

    Returns:
        bool: True if pass the check.
    """
    Pass = True

    # Need to make sure simulation period is longer than a month (GWLF part)
    # Name of keys (Agent and subbasin name) cannot be dulicated.
    # Name of keys (Agent and subbasin name) cannot have "." 
    
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
    Routing = Model["Routing"]
    
    #--- Check In-stream agents:  ResDamAgentTypes & DamDivAgentTypes are eligble in the routing setting. These two types of agents will completely split the system into half.
    InstreamAgentInflows = []        
    InstreamAgentTypes = Model["ABM"]["Inputs"]["ResDamAgentTypes"]+ \
                         Model["ABM"]["Inputs"]["DamDivAgentTypes"]
    for ISagType in InstreamAgentTypes:
        for ag in Model["ABM"][ISagType]:
            Links = Model["ABM"][ISagType][ag]["Inputs"]["Links"]
            InstreamAgentInflows += [outlet for outlet in Links if Links[outlet] == -1 and outlet != ag]
    InstreamAgentInflows = list(set(InstreamAgentInflows))        # Eliminate duplicates.
    #Model["SystemParsedData"]["InStreamAgentInflows"] = InStreamAgentInflows   # Add to system parsed data.

    #--- Check InStreamAgents' inflow outlets are in RoutingOutlets.
    RoutingOutlets = list(Routing.keys())
    RoutingOutlets.remove('Model')  
    for vro in InstreamAgentInflows:
        if vro not in RoutingOutlets:
            logger.error("[Check model failed] Cannot find in-stream agent's inflow outlets in the Routing section. Routing outlets should include {}.".format(vro))
            return False
        else:
            for start in Routing[vro]:
                # Check if start belong to others RoutingOutlets' starts.
                for ro in RoutingOutlets:
                    if ro != vro:
                        if any( start in others for others in Routing[ro] ):
                            if ro in Routing[vro]:  # If it is in the upstream of VirROutlet, it is fine.
                                pass
                            else:
                                logging.error("[Check model failed] {} sub-basin outlet shouldn't be in {}'s (routing outlet) catchment outlets due to the seperation of in-stream control agents.".format(start, vro, ro))
                                return False 
    #--- RiverDivAgentTypes will not add a new routing outlet (agent itself) like ResDamAgentTypes & DamDivAgentTypes do to split the system into half but their diverting outlet must be in  routing outlets.
    RiverDivAgentInflows = []
    for RiverDivType in Model["ABM"]["Inputs"]["RiverDivAgentTypes"]:
        for ag in Model["ABM"][RiverDivType]:
            Links = Model["ABM"][RiverDivType][ag]["Inputs"]["Links"]
            RiverDivAgentInflows += [outlet for outlet in Links if Links[outlet] == -1 and outlet != ag]
    RiverDivAgentInflows = list(set(RiverDivAgentInflows))        # Eliminate duplicates.
    
    #--- Check RiverDivAgents' diverting outlets are in RoutingOutlets.
    for vro in RiverDivAgentInflows:
        if vro not in RoutingOutlets:
            logger.error("[Check model failed] Cannot find RiverDivAgent diverting outlets in the Routing section. Routing outlets should include {}.".format(vro))
            return False    
              
    return True

def parseSimulationSeqence(Model):
    Model["SystemParsedData"]["SimSeq"] = None
    Model["SystemParsedData"]["AgSimSeq"] = None
    Model["SystemParsedData"]["RoutingOutlets"] = None
    Model["SystemParsedData"]["ResDamAgents"] = None
    Model["SystemParsedData"]["DamDivAgents"] = None
    Model["SystemParsedData"]["RiverDivAgents"] = None
    Model["SystemParsedData"]["InStreamAgents"] = None
    Model["SystemParsedData"]["BackTrackingDict"] = None
    Model["SystemParsedData"]["Edges"] = None
    
    #----- Collect in-stream agents
    ## In-stream agents here mean, those agents will redefine the streamflow completely.
    ## which RiverDivAgentTypes, in our definition, only modify the streamflow.
    
    InStreamAgents = []     # Contain only "ResDamAgentTypes" and "DamDivAgentTypes".
    if Model.get("ABM") is not None:
        ABM = Model["ABM"]
        for AgTypes in ["ResDamAgentTypes", "DamDivAgentTypes", "RiverDivAgentTypes"]:    
            Agents = []
            for agType in Model["ABM"]["Inputs"][AgTypes]:
                for end in ABM[agType]:
                    if AgTypes != "RiverDivAgentTypes":     # "RiverDivAgentTypes" since we can multiple ag divert from the same gauge. We modify streamflow directly.
                        InStreamAgents.append(end)
                    Agents.append(end)
            Model["SystemParsedData"][AgTypes[:-5] + "s"] = Agents
    Model["SystemParsedData"]["InStreamAgents"] = InStreamAgents    # Always have InStreamAgents no matter whether ABM section exists.

        
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
                # Eliminate only-sub-basin outlets. If need full stream node sequence, remove this.
                if start in RoutingOutlets + InStreamAgents: 
                    Edges.append((start, end))
                    if BackTrackingDict.get(end) is None:
                        BackTrackingDict[end] = [start]
                    else:
                        BackTrackingDict[end].append(start)
    
    # Add in-stream agents connectinons if ABM sections exists.     
    if Model.get("ABM") is not None:
        InstreamAgentTypes = Model["ABM"]["Inputs"]["ResDamAgentTypes"]+ \
                             Model["ABM"]["Inputs"]["DamDivAgentTypes"]
        ABM = Model["ABM"]
        for agType in InstreamAgentTypes:
            for end in ABM[agType]:
                Links = ABM[agType][end]["Inputs"]["Links"]
                # Since InStreamAgents will completely redefine the streamflow, the inflow factor is defined to be -1.
                # Note that we will not use this factor anywhere in the simulation.
                # end != node condition is designed for DamDivAgentTypes, where its DM is diversion that has factor equal to -1 or < 0 as well.
                # We don't want those edges (e.g. (end, end)) to ruin the simulation sequence formation.
                InflowNodes = [ node for node in Links if Links[node] == -1 and end != node] 
                for start in InflowNodes:
                    Edges.append((start, end))
                    if BackTrackingDict.get(end) is None:
                        BackTrackingDict[end] = [start]
                    else:
                        BackTrackingDict[end].append(start)
    Model["SystemParsedData"]["BackTrackingDict"] = BackTrackingDict
    Model["SystemParsedData"]["Edges"] = Edges
    
    
    # Back tracking to form simulation sequence.        
    def formSimSeq(Node, BackTrackingDict, RoutingOutlets):
        """A recursive function, which keep tracking back upstream nodes until reach the most upstream one. We design this in a clear way. To understand to logic behind, please run step-by-step using the test example at the bottom of the code.
        """
        SimSeq = []
        def trackBack(node, SimSeq, BackTrackingDict, TempDict = {}, AddNode = True):
            if AddNode:
                SimSeq = [node] + SimSeq
            if BackTrackingDict.get(node) is not None:
                routingOutlets = [o for o in BackTrackingDict[node] if o in RoutingOutlets] 
                if len(routingOutlets) >= 1:
                    # g > 1 or len(g) < len(all) => update TempDict
                    if len(routingOutlets) > 1 or len(routingOutlets) < len(BackTrackingDict[node]): 
                        # Get rank of each g
                        rank = []
                        for g in routingOutlets:
                            upList = BackTrackingDict.get(g)
                            if upList is None:
                                rank.append(0)
                            else:
                                rank.append(len(upList))
                        gmax = routingOutlets[rank.index(max(rank))]
                        # Update TempDict: delete node and update others.
                        routingOutlets.remove(gmax)
                        TempDict.pop(gmax, None)    # if 'key' in my_dict: del my_dict['key']
                        for g in routingOutlets:
                            TempDict[g] = node
                        # Call trackBack with gmax and TempDict (recursive)
                        SimSeq, TempDict, BackTrackingDict = trackBack(gmax, SimSeq, BackTrackingDict, TempDict)
                        
                    elif len(routingOutlets) == 1 and len(BackTrackingDict[node]) == 1:
                        SimSeq, TempDict, BackTrackingDict = trackBack(routingOutlets[0], SimSeq, BackTrackingDict, TempDict)
                        TempDict.pop(routingOutlets[0], None)
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
            # elif TempDict != {}:
            #     # Search TempDict and jump backward to add other tributary.
            #     # reverse TempDict
            #     rTempDict = {}
            #     for g in TempDict:
            #         if rTempDict.get(TempDict[g]) is None:
            #             rTempDict[TempDict[g]] = [g]
            #         else:
            #             rTempDict[TempDict[g]].append(g)
            #     if rTempDict != {}:
            #         # Replace BackTrackingDict
            #         for g in rTempDict:
            #             BackTrackingDict[g] = rTempDict[g]
            #         ToNode = SimSeq[min([SimSeq.index(i) for i in rTempDict])]
            #         SimSeq, TempDict, BackTrackingDict = trackBack(ToNode, SimSeq, BackTrackingDict, {}, False)
            return SimSeq, TempDict, BackTrackingDict 
        SimSeq, TempDict, BackTrackingDict = trackBack(Node, SimSeq, BackTrackingDict)
        return SimSeq
    
    
    # LastNode = End nodes - start node. Last node is the only one that does not exist in start nodes.
    if BackTrackingDict == {}:      # If there is only a single outlet!
        SimSeq = RoutingOutlets
    else:
        # end nodes - start nodes = last node. 
        LastNode = list(set(BackTrackingDict.keys()) - set([i[0] for i in Edges]))[0]   
        SimSeq = formSimSeq(LastNode, BackTrackingDict, RoutingOutlets)
    Model["SystemParsedData"]["SimSeq"] = SimSeq
    # Sort RoutingOutlets based on SimSeq
    Model["SystemParsedData"]["RoutingOutlets"] = [ro for ro in SimSeq if ro in RoutingOutlets] 
    #-----------------------------------------------------------------------------------------------
    
    #----- Step2: Form AgSim dictionary -----
    # Aggregate to only SimSeq's node. 
    if Model.get("ABM") is not None:
        ResDamAgentTypes = Model["ABM"]["Inputs"]["ResDamAgentTypes"]
        RiverDivAgentTypes = Model["ABM"]["Inputs"]["RiverDivAgentTypes"]
        DamDivAgentTypes = Model["ABM"]["Inputs"]["DamDivAgentTypes"]
        InsituDivAgentTypes = Model["ABM"]["Inputs"]["InsituDivAgentTypes"]
        RoutingOutlets = Model["SystemParsedData"]["RoutingOutlets"]     # Ordered
        
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
        def createEmptyList(Dict, key):
            if Dict.get(key) is None:
                Dict[key] = []
                
        AgSimSeq = {}              ; Piority = {}
        AgSimSeq["AgSimMinus"] = {}; Piority["AgSimMinus"] = {}
        AgSimSeq["AgSimPlus"] = {} ; Piority["AgSimPlus"] = {}
        for ss in SimSeq:
            AgSimSeq["AgSimMinus"][ss] = {}; Piority["AgSimMinus"][ss] = {}
            AgSimSeq["AgSimPlus"][ss] = {} ; Piority["AgSimPlus"][ss] = {}
        
        for agType in ResDamAgentTypes:
            # Note that in-stream agent (ag) will be in SimSeq
            for ag in Model["ABM"][agType]:
                createEmptyList(AgSimSeq["AgSimPlus"][ag], "ResDamAgents")
                createEmptyList(Piority["AgSimPlus"][ag], "ResDamAgents")
                # No AgSimMinus is needed since in-stream agents replace original streamflow.
                AgSimSeq["AgSimPlus"][ag]["ResDamAgents"].append(ag)
                Piority["AgSimPlus"][ag]["ResDamAgents"].append(0)  # In-stream agent always has piority 0.
                
        for agType in DamDivAgentTypes:
            # Note that in-stream agent (ag) will be in SimSeq
            for ag in Model["ABM"][agType]:
                createEmptyList(AgSimSeq["AgSimPlus"][ag], "DamDivAgents")
                createEmptyList(Piority["AgSimPlus"][ag], "DamDivAgents")
                # No AgSimMinus is needed since in-stream agents replace original streamflow.
                AgSimSeq["AgSimPlus"][ag]["DamDivAgents"].append(ag)
                Piority["AgSimPlus"][ag]["DamDivAgents"].append(0)  # In-stream agent always has piority 0.
        
        for agType in RiverDivAgentTypes:
            for ag in Model["ABM"][agType]:
                Links = Model["ABM"][agType][ag]["Inputs"]["Links"]
                # "list" is our special offer to calibrate return flow factor (Inputs).
                Plus = [p if isinstance(Links[p], list) else p if Links[p] >= 0 else None for p in Links]    
                Minus = [None if isinstance(Links[p], list) else p if Links[p] <= 0 else None for p in Links]
                Plus = list(filter(None, Plus))         # Drop None in a list.
                Minus = list(filter(None, Minus))       # Drop None in a list.
                for p in Plus:
                    ro = searchRoutingOutlet(p)     # Return flow can be added to non-routing outlets. So we need to find the associate routing outlet in the SimSeq.
                    createEmptyList(AgSimSeq["AgSimPlus"][ro], "RiverDivAgents")
                    createEmptyList(Piority["AgSimPlus"][ro], "RiverDivAgents")
                    AgSimSeq["AgSimPlus"][ro]["RiverDivAgents"].append(ag)
                    Piority["AgSimPlus"][ro]["RiverDivAgents"].append(Model["ABM"][agType][ag]["Inputs"]["Piority"])
                for m in Minus:
                    # ro = searchRoutingOutlet(m)  
                    ro = m      # RiverDivAgents diverted outlets should belong to one routing outlet.
                    createEmptyList(AgSimSeq["AgSimMinus"][ro], "RiverDivAgents")
                    createEmptyList(Piority["AgSimMinus"][ro], "RiverDivAgents")
                    AgSimSeq["AgSimMinus"][ro]["RiverDivAgents"].append(ag)
                    Piority["AgSimMinus"][ro]["RiverDivAgents"].append(Model["ABM"][agType][ag]["Inputs"]["Piority"])

        for agType in InsituDivAgentTypes:
            # InsituDivAgentTypes is a simple diversion agent type, which only divert water from a single sub-basin.
            # Runoff of the max(sub-basin - InsituDiv, 0)
            # Note that it divert from runoff of a single sub-basin not river and no return flow option.
            for ag in Model["ABM"][agType]:
                Links = Model["ABM"][agType][ag]["Inputs"]["Links"]
                # No special "list"offer to calibrate return flow factor (Inputs).
                # No return flow option. 
                Minus = [p for p in Links if Links[p] <= 0]
                for m in Minus:
                    ro = searchRoutingOutlet(m)  
                    createEmptyList(AgSimSeq["AgSimMinus"][ro], "InsituDivAgents")
                    createEmptyList(Piority["AgSimMinus"][ro], "InsituDivAgents")
                    AgSimSeq["AgSimMinus"][ro]["InsituDivAgents"].append(ag)
                    Piority["AgSimMinus"][ro]["InsituDivAgents"].append(Model["ABM"][agType][ag]["Inputs"]["Piority"])
                    
        # Sort agents based on their piorities               
        for pm in AgSimSeq:
            for AgTypes in AgSimSeq[pm]:
                for ro in AgSimSeq[pm][AgTypes]:
                    Agents = AgSimSeq[pm][AgTypes][ro]
                    Piorities = Piority[pm][AgTypes][ro]
                    Agents = [ag for _,ag in sorted(zip(Piorities,Agents))]
                    AgSimSeq[pm][AgTypes][ro] = list(set(Agents))    # Remove duplicated ags.

        Model["SystemParsedData"]["AgSimSeq"] = AgSimSeq    
        #----------------------------------------
    SummaryDict = {}
    for i in ["SimSeq","RoutingOutlets","ResDamAgents","DamDivAgents","RiverDivAgents","InStreamAgents","AgSimSeq"]:
        SummaryDict[i] = Model["SystemParsedData"][i]
    ParsedModelSummary = Dict2String(SummaryDict, Indentor = "  ")
    logger.info("Parsed model data summary:\n" + ParsedModelSummary)
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
RoutingOutlets = ["g1","g2","g3","g4","g5","g6","g7","g8","G"]
Node = "G"


SimSeq = formSimSeq(Node, BackTrackingDict, RoutingOutlets)
# Expect to get ['g4', 'g5', 'g1', 'g2', 'g3', 'g6', 'V1', 'R1', 'g7', 'g8', 'G']

"""