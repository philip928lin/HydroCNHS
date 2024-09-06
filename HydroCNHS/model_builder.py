# Model builder module.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com)
# Last update at 2022/5/19.
import os
from copy import deepcopy
import pandas as pd
from .util import write_model, dict_to_string
from .abm_script import *

model_template = {
    "Path": {"WD": "", "Modules": ""},
    "WaterSystem": {
        "StartDate": "yyyy/mm/dd",
        "EndDate": "yyyy/mm/dd",
        "DataLength": None,
        "NumSubbasins": None,
        "Outlets": [],
        "NodeGroups": [],
        "RainfallRunoff": None,
        "Routing": "Lohmann",
        "ABM": None,
    },
    "RainfallRunoff": {},
    "Routing": {},
}

GWLF_template = {
    "Inputs": {"Area": None, "Latitude": None, "S0": 2, "U0": 10, "SnowS": 5},
    "Pars": {
        "CN2": -99,
        "IS": -99,
        "Res": -99,
        "Sep": -99,
        "Alpha": -99,
        "Beta": -99,
        "Ur": -99,
        "Df": -99,
        "Kc": -99,
    },
}

ABCD_template = {
    "Inputs": {"Area": None, "Latitude": None, "XL": 2, "SnowS": 5},
    "Pars": {"a": -99, "b": -99, "c": -99, "d": -99, "Df": -99},
}

Other_template = {"Inputs": {}, "Pars": {}}

Lohmann_template = {
    "Inputs": {"FlowLength": None, "InstreamControl": False},
    "Pars": {"GShape": -99, "GScale": -99, "Velo": -99, "Diff": -99},
}

ABM_template = {
    "Modules": [],
    "InstitDMClasses": {},  # {InstitDMClasses: Institution}
    "DMClasses": [],
    "DamAPI": [],
    "RiverDivAPI": [],
    "InsituAPI": [],
    "ConveyingAPI": [],
    "Institutions": {},  # {Institution: [agent list]}
}

agent_template = {
    "Attributes": {},
    "Inputs": {"Priority": 0, "Links": {}, "DMClass": None},
    "Pars": {},
}


class ModelBuilder(object):
    def __init__(self, wd):
        self.model = deepcopy(model_template)
        self.model["Path"]["WD"] = wd
        self.wd = wd
        self.help()
        print("Use .help to re-print the above instructions.")

        class APIs(object):
            def __init__(self):
                self.Dam = "DamAPI"
                self.InSitu = "InSituAPI"
                self.RiverDiv = "RiverDivAPI"
                self.Conveying = "ConveyingAPI"

        self.api = APIs()

    def help(self):
        print(
            "Follow the following steps to create model & ABM script templates:\n"
            + "\tStep 1: set_water_system()\n"
            + "\tStep 2: set_rainfall_runoff()\n"
            + "\tStep 3: set_routing_outlet(), one at a time.\n"
            + "\tStep 4: (optional) set_ABM().\n"
            + "\tStep 5: (optional) add_agent().\n"
            + "\tStep 6: (optional) add_institution().\n"
            + "\tStep 7: write_model_to_yaml()\n"
            + "\tStep 8: gen_ABM_script_template()\n"
            + "Open generated model.yaml & ABM module template "
            + "and further edit them."
        )

    def set_water_system(self, start_date, end_date):
        """Set up WaterSystem.

        Parameters
        ----------
        start_date : str
            "yyyy/mm/dd"
        end_date : str
            "yyyy/mm/dd"
        """
        self.model["WaterSystem"]["StartDate"] = start_date
        self.model["WaterSystem"]["EndDate"] = end_date
        end_date = pd.to_datetime(end_date, format="%Y/%m/%d")
        start_date = pd.to_datetime(start_date, format="%Y/%m/%d")
        self.model["WaterSystem"]["DataLength"] = (end_date - start_date).days + 1

    def set_rainfall_runoff(
        self, outlet_list, area_list=None, lat_list=None, runoff_model="GWLF"
    ):
        """Set up RainfallRunoff.

        Parameters
        ----------
        outlet_list : list
            A list of subbasin outlet names.
        area_list : list, optional
            Area [ha] list corresponding to outlet_list, by default None.
        lat_list : str, optional
            Latitude [deg] list corresponding to outlet_list, by default None.
        runoff_model : str, optional
            "GWLF" or "ABCD" or "Other", by default None.

        Note :
            If "Other" is selected for runoff_model, users must provide
            precalculated runoffs for each subbasin as an input to HydroCNHS.
        """
        self.model["WaterSystem"]["Outlets"] = outlet_list
        self.model["WaterSystem"]["NumSubbasins"] = len(outlet_list)
        self.model["WaterSystem"]["RainfallRunoff"] = runoff_model

        # Select RainfallRunoff templete.
        if runoff_model == "GWLF":
            RainfallRunoff_templete = GWLF_template
        elif runoff_model == "ABCD":
            RainfallRunoff_templete = ABCD_template
        elif runoff_model == "Other":
            RainfallRunoff_templete = Other_template
        else:
            raise ValueError(
                "Given rainfall-runoff model, {}, is not eligible.".format(runoff_model)
            )

        for sub in outlet_list:
            self.model["RainfallRunoff"][sub] = deepcopy(RainfallRunoff_templete)

        if area_list is not None:
            for i, sub in enumerate(outlet_list):
                self.model["RainfallRunoff"][sub]["Inputs"]["Area"] = area_list[i]
        if lat_list is not None:
            for i, sub in enumerate(outlet_list):
                self.model["RainfallRunoff"][sub]["Inputs"]["Latitude"] = lat_list[i]

    def set_routing_outlet(
        self,
        routing_outlet,
        upstream_outlet_list,
        instream_objects=[],
        flow_length_list=None,
        routing_model="Lohmann",
    ):
        """Set up a routing outlet.

        Parameters
        ----------
        routing_outlet : str
            Name of routing outlet. routing_outlet should be one of outlets in
            RainfallRunoff.
        upstream_outlet_list : list
            A list of outlets or dam agents that contribute to the streamflow
            at routing_outlet.
        instream_objects : list, optional
            A list of instream objects' names (i.e., dam agents), by default [].
        flow_length_list : list, optional
            A list of flow lengths. The order has to consist to the
            upstream_outlet_list.
        routing_model : list, optional
            Routing model, by default "Lohmann".

        """
        outlet_list = self.model["WaterSystem"]["Outlets"]
        self.model["WaterSystem"]["Routing"] = routing_model

        if routing_outlet not in outlet_list:
            raise ValueError(
                "Given routing_outlet, {}, ".format(routing_outlet)
                + "is not in the outlet list {}.".format(str(outlet_list))
                + "Please run set_runoff() first."
            )

        if flow_length_list is None:
            flow_length_list = [None] * len(upstream_outlet_list)

        if routing_outlet not in upstream_outlet_list:
            upstream_outlet_list.append(routing_outlet)
            flow_length_list.append(0)

        self.model["Routing"][routing_outlet] = {}
        route_ro = self.model["Routing"][routing_outlet]

        for i, o in enumerate(upstream_outlet_list):
            if o in instream_objects:
                # No within-subbasin routing.
                route_ro[o] = deepcopy(Lohmann_template)
                route_ro[o]["Pars"]["GShape"] = None
                route_ro[o]["Pars"]["GScale"] = None
                route_ro[o]["Inputs"]["InstreamControl"] = True
            elif o not in outlet_list:
                raise ValueError(
                    "Given upstream outlet, {}, ".format(o)
                    + "is not in the outlet list {}.".format(str(outlet_list))
                    + "Please run set_runoff() first."
                )
            elif o == routing_outlet:
                # No inter-subbasin routing.
                route_ro[o] = deepcopy(Lohmann_template)
                route_ro[o]["Pars"]["Velo"] = None
                route_ro[o]["Pars"]["Diff"] = None
            else:
                route_ro[o] = deepcopy(Lohmann_template)

            route_ro[o]["Inputs"]["FlowLength"] = flow_length_list[i]

        # Turn of within-subbasin routing if upstream outlets of other routing
        # are routing_outlet (this newly added one).
        routing_outlets = list(self.model["Routing"].keys())
        for ro in routing_outlets:
            for o in list(self.model["Routing"][ro].keys()):
                if o in routing_outlets and o != ro:
                    self.model["Routing"][ro][o]["Pars"]["GShape"] = None
                    self.model["Routing"][ro][o]["Pars"]["GScale"] = None

    def set_ABM(self, abm_module_folder_path=None, abm_module_name="ABM_module.py"):
        """Set up ABM

        Parameters
        ----------
        abm_module_folder_path : str, optional
            Folder directory of ABM modules. It it is not given, working
            directory will be assigned, by default None.
        abm_module_name : str, optional
            The ABM module name, by default "ABM_module.py"
        """
        if abm_module_folder_path is None:
            abm_module_folder_path = self.wd
        if abm_module_name[-3:] != ".py":
            abm_module_name = abm_module_name + ".py"

        self.abm_module_folder_path = abm_module_folder_path
        self.abm_module_name = abm_module_name
        self.model["WaterSystem"]["ABM"] = deepcopy(ABM_template)
        self.model["WaterSystem"]["ABM"]["Modules"].append(abm_module_name)
        self.model["Path"]["Modules"] = abm_module_folder_path
        self.model["ABM"] = {}

    def add_agent(
        self,
        agt_type_class,
        agt_name,
        api,
        priority=1,
        link_dict={},
        dm_class=None,
        par_dict={},
        attr_dict={},
    ):
        """Add agent.

        Parameters
        ----------
        agt_type_class : str
            Assigned agent type class.
        agt_name : str
            Agent name.
        api : str
            The API to integrate the agent to the HydroCNHS.
            e.g., mb.Dam.
        priority : int, optional
            Priority of the agent if conflicts occur, by default 1.
        link_dict : dict, optional
            Linkage dictionary, by default {}.
        dm_class : str, optional
            Assigned decision-making class, by default None
        par_dict : dict, optional
            Parameter dictionary, by default {}
        attr_dict : dict, optional
            Attribution dictionary, by default {}
        """
        if self.model.get("ABM") is None:
            raise KeyError("ABM has not been set. Please run 'set_ABM' first.")

        abm = self.model["ABM"]
        if abm.get(agt_type_class) is None:
            abm[agt_type_class] = {}

        ws_abm = self.model["WaterSystem"]["ABM"]
        if api == self.api.Dam:
            ws_abm["DamAPI"].append(agt_type_class)
            ws_abm["DamAPI"] = list(set(ws_abm["DamAPI"]))
            Priority = 0
        elif api == self.api.RiverDiv:
            ws_abm["RiverDivAPI"].append(agt_type_class)
            ws_abm["RiverDivAPI"] = list(set(ws_abm["RiverDivAPI"]))
        elif api == self.api.Conveying:
            ws_abm["ConveyingAPI"].append(agt_type_class)
            ws_abm["ConveyingAPI"] = list(set(ws_abm["ConveyingAPI"]))
        elif api == self.api.InSitu:
            ws_abm["InsituAPI"].append(agt_type_class)
            ws_abm["InsituAPI"] = list(set(ws_abm["InsituAPI"]))
        else:
            raise KeyError(
                "Assigned api is not eligible. Eligible apis "
                + "including {}, {}, {}, and {}".format(
                    self.api.Dam, self.api.RiverDiv, self.api.Conveying, self.api.InSitu
                )
            )

        ws_abm["DMClasses"].append(dm_class)
        ws_abm["DMClasses"] = list(set(ws_abm["DMClasses"]))

        abm[agt_type_class][agt_name] = deepcopy(agent_template)
        abm[agt_type_class][agt_name]["Attributes"] = deepcopy(attr_dict)
        abm[agt_type_class][agt_name]["Inputs"]["Priority"] = priority
        abm[agt_type_class][agt_name]["Inputs"]["Links"] = deepcopy(link_dict)
        abm[agt_type_class][agt_name]["Inputs"]["DMClass"] = dm_class
        abm[agt_type_class][agt_name]["Pars"] = deepcopy(par_dict)

    def add_institution(self, institution, instit_dm_class, agent_list):
        """Add a institution.

        Parameters
        ----------
        institution : str
            Institution name.
        instit_dm_class : str
            Assigned institutional decision-making class.
        agent_list : list
            Agent member list of the institute.
        """
        if self.model.get("ABM") is None:
            raise KeyError("ABM has not been set. Please run 'set_ABM' first.")

        abm = self.model["ABM"]
        for _, agts_dict in abm.items():
            for agt, agt_config in agts_dict.items():
                if agt in agent_list:
                    agt_config["Inputs"]["DMClass"] = institution

        instit_dm = self.model["WaterSystem"]["ABM"]["InstitDMClasses"]
        if instit_dm.get(instit_dm_class) is None:
            instit_dm[instit_dm_class] = []
        instit_dm[instit_dm_class].append(institution)

        instits = self.model["WaterSystem"]["ABM"]["Institutions"]
        instits[institution] = agent_list

        # Remove from DMClass if exist
        if institution in self.model["WaterSystem"]["ABM"]["DMClasses"]:
            self.model["WaterSystem"]["ABM"]["DMClasses"].remove(institution)

    def write_model_to_yaml(self, filename):
        """Output model configuration file (.yaml)

        Parameters
        ----------
        filename : str
            Filename
        """
        if filename[-5:] != ".yaml":
            filename = filename + ".yaml"
        filename = os.path.join(self.wd, filename)
        write_model(self.model, filename)
        print(
            "Model configuration file (.yaml) have been save at {}.".format(filename),
            "Please open the file and further edit it.",
        )

    def gen_ABM_module_template(self):
        """Generate ABM module template based on the ABM setting."""
        filename = os.path.join(self.abm_module_folder_path, self.abm_module_name)
        ws_abm = self.model["WaterSystem"]["ABM"]
        abm = self.model["ABM"]
        with open(filename, "w") as f:
            f.write(import_str + design_str)
            for agt_type in list(abm.keys()):
                f.write(add_agt_class(agt_type))
            for dm_class in ws_abm["DMClasses"]:
                f.write(add_dm_class(dm_class, is_institution=False))
            for intit_dm_class in list(ws_abm["InstitDMClasses"].keys()):
                f.write(add_dm_class(intit_dm_class, is_institution=True))
        print(
            "ABM module template (.py) have been save at {}.".format(filename),
            "Please open the file and further edit it.",
        )

    def print_model(self, indentor="  ", level=1):
        """Print model to the console

        Parameters
        ----------
        indentor : str, optional
            Indentor, by default "  ".
        level : int, optional
            Print out level of a nested dictionary, by default 1.
        """
        print(dict_to_string(self.model, indentor, level))
