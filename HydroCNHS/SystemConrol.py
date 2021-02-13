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
    with open(model, 'rt') as file: 
        try:
            Model = yaml.safe_load(file.read())
            checkModel(Model)
        except Exception as e:
            logger.error(traceback.format_exc())   # Logs the error appropriately.
            return None
    
    # Fill VirROutlets for routing if ABM and Routing exists.
    if Model.get("Routing") is not None and Model.get("ABM") is not None:
        InStreamAgentTypes = Model["ABM"]["Inputs"]["InStreamAgentTypes"]
        VirROutlets = []
        for ISagType in InStreamAgentTypes:
            for ag in Model["ABM"][ISagType]:
                Links = Model["ABM"][ISagType][ag]["Inputs"]["Links"]
                VirROutlets += [outlet for outlet in Links if Links[outlet] == -1]
        VirROutlets = list(set(VirROutlets))    # Eliminate duplicates.
        Model["ABM"]["Inputs"]["VirROutlets"] = VirROutlets
        
    # Check model is consist and correct.
    if Checked:     # We don't recheck model is, it has been checked.
        return Model
    else:
        if checkModel(Model):
            return Model
        else:
            return None
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
#---------- Check Functions ----------
def checkModel(Model):
    """Check the consistency of the model dictionary.

    Args:
        model (dict): Loaded from model.yaml. 

    Returns:
        bool: True if pass the check.
    """
    Pass = True
    Pass = checkInStreamAgentInRouting(Model)
    return Pass

def checkInStreamAgentInRouting(Model):
    GaugedOutlets = Model["WaterSystem"]["GaugedOutlets"]
    Routing = Model["Routing"]
    VirROutlets = Model["ABM"]["Inputs"]["VirROutlets"]

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
                            
#-------------------------------------

#%% Test
r"""
from pprint import pprint
pprint(loadConfig())
initialize(WD = r"C:\Users\Philip\OneDrive\Lehigh\0_Proj2_UA-SA-Equifinality\ModelRunTest")
"""