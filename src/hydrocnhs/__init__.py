# Load functions that directly available for user when the package is loaded.
from .util import (
    load_model,
    write_model,
    write_model_to_df,
    write_model_to_csv,
    load_df_to_model_dict,
    create_rn_gen,
    set_logging_config,
    gen_default_bounds,
)
from .hydrocnhs import Model
from .model_builder import ModelBuilder
from .data_collector import Data_collector
from .indicators import Indicator
from .visual import Visual

# from .rainfall_runoff_model.pet_hamon import cal_pet_Hamon

# Setup logging when HydroCNHS is imported.
# Default only show console log. Log file is created in the HydroCNHS class
# according to user's setting.
set_logging_config()
