from .SystemConrol import loadLoggingConfig
# Setup logging when HydroCNHS is imported.
# Default only show console log. Log file is created in the HydroCNHS class according to user's setting.
loadLoggingConfig()