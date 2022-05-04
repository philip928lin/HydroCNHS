# Model builder module.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# Last update at 2021/12/22.

from copy import deepcopy
import pandas as pd
from .util import write_model, dict_to_string

model_template = {
    "Path": {"WD": "",
             "Modules": ""},
    "WaterSystem": {
        "StartDate": "yyyy/mm/dd",
        "EndDate": "yyyy/mm/dd",
        "NumSubbasins": None,
        "NumGauges": None,
        "NumAgents": None,
        "Outlets": [],
        "GaugedOutlets": [],
        "DataLength": None
    },
    "LSM": {
        "Model": ""
    },
    "Routing": {
        "Model": "Lohmann",
    }
}

GWLF_template = {
    "Inputs": {"Area":      None,
               "Latitude":  None,
               "S0":        None,
               "U0":        None,
               "SnowS":     None},
    "Pars": {"CN2":     -99,
             "IS":      -99,
             "Res":     -99,
             "Sep":     -99,
             "Alpha":   -99,
             "Beta":    -99,
             "Ur":      -99,
             "Df":      -99,
             "Kc":      -99}
}

ABCD_template = {
    "Inputs": {"Area":      None,
               "Latitude":  None,
               "XL":        None,
               "SnowS":     None},
    "Pars": {"a":     -99,
             "b":      -99,
             "c":     -99,
             "d":     -99,
             "Df":      -99}
}

Lohmann_template = {
    "Inputs": {"FlowLength":        None,
               "InstreamControl":   False},
    "Pars": {"GShape":   -99,
             "GScale":   -99,
             "Velo":     -99,
             "Diff":     -99}
}

ABM_template = {
    "Inputs": {
        "DamAgentTypes": [],
        "RiverDivAgentTypes": [],
        "InsituAgentTypes": [],
        "ConveyAgentTypes": [],
        "DMClasses": [],
        "Modules": [],
        "AgGroup": None
    }
}

class ModelBuilder(object):
    def __init__(self, wd):
        self.Model = deepcopy(model_template)
        self.help()
        print("Use .help to re-print the above instruction.")
        
    def help(self):
        print("Follow the following steps to create model template:\n"
              +"\tStep 1: set_water_system()\n"
              +"\tStep 2: set_lsm()\n"
              +"\tStep 3: set_routing_outlet(), one at a time.\n"
              +"\tStep 4: set_ABM() if you want to build a coupled model.\n"
              +"\tStep 5: write_model_to_yaml()\n"
              +"After creating model.yaml template, "
              +"open it and further edit it.")
    def set_water_system(self, start_date, end_date):
        """Setup WaterSystem.

        Parameters
        ----------
        start_date : str
            "yyyy/mm/dd"
        end_date : str
            "yyyy/mm/dd"
        """
        self.Model["WaterSystem"]["StartDate"] = start_date
        self.Model["WaterSystem"]["EndDate"] = end_date
        end_date = pd.to_datetime(end_date, format='%Y/%m/%d')
        start_date = pd.to_datetime(start_date, format='%Y/%m/%d')
        self.Model["WaterSystem"]["DataLength"] = (end_date-start_date).days+1
    
    def set_lsm(self, outlet_list, lsm_model="GWLF"):
        """Setup LSM.

        Parameters
        ----------
        outlet_list : list
            List of outlet names.
        lsm_model : str, optional
            "GWLF" or "ABCD", by default "GWLF"
        """
        self.Model["WaterSystem"]["Outlets"] = outlet_list
        self.Model["WaterSystem"]["NumSubbasins"] = len(outlet_list)
        
        # Select LSM templete.
        if lsm_model == "GWLF":
            LSM_templete = GWLF_template
        elif lsm_model == "ABCD":
            LSM_templete = ABCD_template
        else:
            raise ValueError(
                "Given LSM model, {}, is not eligible.".format(lsm_model))
        
        self.Model["LSM"]["Model"] = lsm_model
        for sub in outlet_list:
            self.Model["LSM"][sub] = deepcopy(LSM_templete)                                 
    
    def set_routing_outlet(self, routing_outlet, upstream_outlet_list,
                           instream_outlets=[]):
        """Setup routing outlet one by one.

        Parameters
        ----------
        routing_outlet : str
            Name of routing outlet. routing_outlet should be one of outlets in
            LSM.
        upstream_outlet_list : list
            A list of outlets or dam agents that contribute to the streamflow
            at routing_outlet.
        instream_outlets : list, optional
            A list of instream outlets' names (i.e., dam agents), by default [].

        """
        outlet_list = self.Model["WaterSystem"]["Outlets"]
        
        if routing_outlet not in outlet_list:
            raise ValueError(
                "Given routing_outlet, {}, ".format(routing_outlet)
                +"is not in the outlet list {}.".format(str(outlet_list))
                +"Please run set_lsm() first.")
        
        self.Model["Routing"][routing_outlet] = {}
        route_ro = self.Model["Routing"][routing_outlet]
        
        for o in upstream_outlet_list:
            if o in instream_outlets:
                # No within-subbasin routing.
                route_ro[o] = deepcopy(Lohmann_template)
                route_ro[o]["Pars"]["GShape"] = None
                route_ro[o]["Pars"]["GScale"] = None
            elif o not in outlet_list:
                raise ValueError(
                    "Given upstream outlet, {}, ".format(o)
                    +"is not in the outlet list {}.".format(str(outlet_list))
                    +"Please run set_lsm() first.")
            elif o == routing_outlet:
                # No inter-subbasin routing.
                route_ro[o] = deepcopy(Lohmann_template)
                route_ro[o]["Pars"]["Velo"] = None
                route_ro[o]["Pars"]["Diff"] = None
            else:
                route_ro[o] = deepcopy(Lohmann_template)
        
        # Turn of within-subbasin routing if upstream outlets of other routing 
        # are routing_outlet (this newly added one).
        routing_outlets = list(self.Model["Routing"].keys())
        for ro in routing_outlets:
            if ro != "Model":
                for o in list(self.Model["Routing"][ro].keys()):
                    if o in routing_outlets and o != ro:
                        self.Model["Routing"][ro][o]["Pars"]["GShape"] = None
                        self.Model["Routing"][ro][o]["Pars"]["GScale"] = None
                    
    def set_ABM(self, abm_module_path=""):
        """Setup ABM if it is a coupled model.
        
        Please manually setup the rest of ABM setting by editing .yaml file
        directly following our tutorial.
        Note ABM is not required if the model is not a coupled model.
        """
        self.Model["ABM"] = deepcopy(ABM_template)
        self.Model["Path"]["Modules"] = abm_module_path
        
    def write_model_to_yaml(self, filename):
        if filename[-5:] != ".yaml":
            filename = filename + ".yaml"
        write_model(self.Model, filename)
        print("Save at {}.".format(filename))
    
    def print_model(self):
        print(dict_to_string(self.Model))

r"""
import HydroCNHS
m = HydroCNHS.ModelBuilder("WD")
m.set_water_system("1980/8/5", "2050/4/7")
m.set_lsm(["a", "b", "c", "d"], "ABCD")
m.set_routing_outlet("b", ["c", "d", "b"], ["c"])
m.set_routing_outlet("d", ["d"])
m.set_ABM(abm_module_path="XDD")
m.write_model_to_yaml(r"C:\Users\model_test.yaml")
m.print_model()
"""