# Load functions that directly available for user when the package is loaded.  
from .SystemConrol import loadLoggingConfig, updateConfig, defaultConfig, loadModel
from .HydroCNHS import HydroCNHS
# Setup logging when HydroCNHS is imported.
# Default only show console log. Log file is created in the HydroCNHS class according to user's setting.
loadLoggingConfig()


