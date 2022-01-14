import HydroCNHS

wd = "working directory"
mb = HydroCNHS.ModelBuilder(wd)

r'''
Follow the following steps to create model template:
	Step 1: set_water_system()
	Step 2: set_lsm()
	Step 3: set_routing_outlet(), one at a time.
	Step 4: set_ABM() if you want to build a coupled model.
	Step 5: write_model_to_yaml()
After creating model.yaml template, open it and further edit it.
Use .help to re-print the above instruction.
'''

### Setup a water system simulation information
mb.set_water_system(start_date="1981/1/1", end_date="2013/12/31")

### Setup land surface model (rainfall-runoff model)
# Here we have seven subbasins and we select GWLF as the rainfall-runoff model.
outlet_list = ["TRTR", "HaggIn", "DLLO", "TRGC", "DAIRY", "RCTV", "WSLO"]
mb.set_lsm(outlet_list=outlet_list, lsm_model="GWLF")   # or lsm_model="ABCD"

### Setup routing 
# We have four routing outlets, which we will add them into the model one by 
# one.
mb.set_routing_outlet(routing_outlet="WSLO", 
                      upstream_outlet_list=["TRGC", "DAIRY", "RCTV", "WSLO"])
mb.set_routing_outlet(routing_outlet="TRGC", 
                      upstream_outlet_list=["DLLO", "TRGC"])
mb.set_routing_outlet(routing_outlet="DLLO", 
                      upstream_outlet_list=["ResAgt", "TRTR", "DLLO"], 
                      instream_outlets=["ResAgt"]) 
# Note: ResAgt (Hagg Lake) is the reservoir agent, whcih is considerred as an 
# instream control object.

mb.set_routing_outlet(routing_outlet="HaggIn", 
                      upstream_outlet_list=["HaggIn"])

### Setup ABM
abm_module_path="abm_module path"
mb.set_ABM(abm_module_path=abm_module_path)

### Save to a .yaml model file for further editting.
filename = "output directory/model.yaml"
mb.write_model_to_yaml(filename)