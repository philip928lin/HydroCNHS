#%%
# System control file for HydroCNHS.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# 2021/02/05

import logging
import logging.config
import traceback
import yaml
import os 

r"""
We need to modify yaml, which we can load and write the file while keeping comments.
https://stackoverflow.com/questions/7255885/save-dump-a-yaml-file-with-comments-in-pyyaml/27103244
"""
this_dir, this_filename = os.path.split(__file__)
def loadConfig():
    """Get config dictionary from Config.yaml.

    Returns:
        dict: Dictionary of model config.
    """
    print(os.path.join(this_dir, 'Config.yaml'))
    with open(os.path.join(this_dir, 'Config.yaml'), 'rt') as file:
        config = yaml.safe_load(file.read())
    return config

def loadLoggingConfig():
    """Load logging configuration and setup logging.
    """
    Config = loadConfig()
    with open(os.path.join(this_dir, 'LoggingConfig.yaml'), 'rt') as file:
        LoggingConfig = yaml.safe_load(file.read())
    if Config["LogHandlers"] is not None:       # Customize log msg to console/log file/both
        LoggingConfig["loggers"]["HydroCNHS"]["handlers"] = Config["LogHandlers"]
    logging.config.dictConfig(LoggingConfig)

# def writeConfig(Config):
#     """Write Config.yaml.

#     Args:
#         Config (dict): Dictionary of model config
#     """
#     with open('Config.yaml', 'w') as file:
#         yaml.dump(Config, file)

# def defaultConfig():
#     """Default Config.yaml.
#     """
#     with open('Config_default.yaml', 'rt') as file:
#         Config_default = yaml.safe_load(file.read())
#     with open('Config.yaml', 'w') as file:
#         yaml.dump(Config_default, file)
    
def loadModel(model):
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
    # Check model is consist and correct.
    
    return Model

def checkModel(model):
    """Check the consistency of the model dictionary.

    Args:
        model (dict): Loaded from model.yaml. 

    Returns:
        bool: True if pass the check.
    """
    return True
#%% Test
r"""
from pprint import pprint
pprint(loadConfig())
initialize(WD = r"C:\Users\Philip\OneDrive\Lehigh\0_Proj2_UA-SA-Equifinality\ModelRunTest")
"""