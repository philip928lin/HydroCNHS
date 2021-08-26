import os
import yaml
import logging
import logging.config 

# Load functions that directly available for user when the package is loaded.
from .util import (load_system_config, update_system_config, default_config,
                   load_model, write_model, write_model_to_df,
                   write_model_to_csv, load_df_to_model_dict, set_seed)
from .hydrocnhs import HydroCNHSModel
from .model_builder import ModelBuilder
from .indicators import Indicator
from .visual import Visual
from .land_surface_model.pet_hamon import cal_pet_Hamon

# Setup logging when HydroCNHS is imported.
# Default only show console log. Log file is created in the HydroCNHS class
# according to user's setting.
this_dir, this_filename = os.path.split(__file__)
def load_logging_config():
    """Load logging configuration and setup logging.
    """
    config = load_system_config()
    with open(os.path.join(this_dir, 'Config_logging.yaml'), 'rt') as file:
        logging_config = yaml.safe_load(file.read())
    # Customize log msg to console/log file/both
    if config["LogHandlers"] is not None:
        logging_config["loggers"]["HydroCNHS"]["handlers"] = \
            config["LogHandlers"]
    logging.config.dictConfig(logging_config)
load_logging_config()


