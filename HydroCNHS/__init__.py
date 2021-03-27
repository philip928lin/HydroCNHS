# Load functions that directly available for user when the package is loaded.
from .SystemConrol import loadConfig, updateConfig, defaultConfig, loadModel, writeModel, writeModelToDF, writeModelToCSV, loadDFToModelDict, setSeed
from .HydroCNHS import HydroCNHSModel
from .Calibration.Calibration import Cali
from .ModelBuilder import ModelBuilder
from .Indicators import Indicator
from .Plot import Plot

# Setup logging when HydroCNHS is imported.
# Default only show console log. Log file is created in the HydroCNHS class according to user's setting.
import logging
import logging.config  
import yaml
import os
this_dir, this_filename = os.path.split(__file__)
def loadLoggingConfig():
    """Load logging configuration and setup logging.
    """
    Config = loadConfig()
    with open(os.path.join(this_dir, 'LoggingConfig.yaml'), 'rt') as file:
        LoggingConfig = yaml.safe_load(file.read())
    if Config["LogHandlers"] is not None:       # Customize log msg to console/log file/both
        LoggingConfig["loggers"]["HydroCNHS"]["handlers"] = Config["LogHandlers"]
    logging.config.dictConfig(LoggingConfig)
loadLoggingConfig()


