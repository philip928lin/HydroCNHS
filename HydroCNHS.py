import logging
import logging.config
import yaml
import os 

def getConfig():
    """Get config dictionary from Config.yaml.

    Returns:
        dict: Dictionary of model config.
    """
    with open('HydroCNHS\Config.yaml', 'rt') as file:
        config = yaml.safe_load(file.read())
    return config

def writeConfig(Config):
    """Write Config.yaml.

    Args:
        Config (dict): Dictionary of model config
    """
    with open('HydroCNHS\Config2.yaml', 'w') as file:
        yaml.dump(Config, file)

def initialize(WD):
    """Initialize HydroCNHS model.
    Assign the WD to the config file and initialize the logging setting.

    Args:
        WD (string): Working directory
    """
    Config = getConfig()            # Get Config
    Config["Path"]["WD"] = WD       # Assign working directory
    writeConfig(Config)             # Write Config
    
    # Initialize logging setting
    with open('LoggingConfig.yaml', 'rt') as file:
        LoggingConfig = yaml.safe_load(file.read())
    if WD is not None:              # Assign log file output path to logging
        LoggingConfig["handlers"]["file"]["filename"] = os.path.join(WD, "HydroCNHS.log")
    logging.config.dictConfig(LoggingConfig)
    # Get logger and start the logging process
    logger = logging.getLogger("HydroCNHS")
    logger.info("Welcome to HydroCNHS!")