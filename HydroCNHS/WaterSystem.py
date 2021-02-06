#---- Loading input ----------
# Load Config
# Load model
with open('Config.yaml', 'rt') as file:
        config = yaml.safe_load(file.read())
        WD = config["Path"]["WD"]
        print(config)
