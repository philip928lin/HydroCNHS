# Primary HydroCNHS simulator.
# This file control the coupling logic of CHNS model.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com)
# 2021/02/05.
# Last update at 2022/1/16.

import time
import traceback
from copy import Error, deepcopy
import numpy as np
from pandas import date_range, to_datetime
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
from .rainfall_runoff_model.abcd import run_ABCD
from .rainfall_runoff_model.gwlf import run_GWLF
from .rainfall_runoff_model.pet_hamon import cal_pet_Hamon
from .routing import form_UH_Lohmann, run_step_Lohmann, run_step_Lohmann_convey
from .util import (set_logging_config, load_model,
                   load_customized_module_to_class,
                   list_callable_public_object)
from .data_collector import Data_collector

class Model(object):
    def __init__(self, model, name=None, rn_gen=None, checked=False,
                 parsed=False, log_filename=None):
        """Create a HydroCNHS model.

        Parameters
        ----------
        model : dict/str
            HydroCNHS model. It can be provided by a dictionary or .yaml file
            name.
        name : str, optional
            The name of the created model, by default None.
        rn_gen : object, optional
            Random number generator created by create_rn_gen(), by default None.
            If given, randomness of the designed model is controled by rn_gen.
            We encourage user to assign it to maintain the reproducibility of
            the stochastic simulation.
        checked : bool, optional
            If true, no checking process will be conducted, by default False.
        parsed : bool, optional
            If true, the model will not be re-parsed, by default False.
        log_filename : If log filename is given, a log file will be created, by
            default None. Note: Do not create the log file when calbrating the
            model in parallel. Unexpected I/O errors may occur.
        """
        # Assign model name and get logger.
        self.name = name
        if log_filename is not None:
            set_logging_config(log_filename)

        # Get logger.
        if name is None:
            self.logger = logging.getLogger("HydroCNHS")
        else:
            self.logger = logging.getLogger("HydroCNHS."+name)
        logger = self.logger

        # Parallelization setting
        self.paral_setting = {"verbose": 0,
                              "cores_formUH": -1,
                              "cores_runoff": -1}
        # Load model.yaml
        model = load_model(model, checked=checked, parsed=parsed)

        # Create random number generator for ABM.
        if rn_gen is None:
            # Assign a random seed.
            seed = np.random.randint(0, 100000)
            self.rn_gen = np.random.default_rng(seed)
        else:
            # User-provided rn generator
            self.rn_gen = rn_gen
            self.ss = rn_gen.bit_generator._seed_seq
            logger.info(
                "User-provided random number generator is assigned.")

        # Verify model contain all following keys.
        try:
            self.path = model["Path"]
            self.ws = model["WaterSystem"]  # ws: Water system
            self.runoff = model["RainfallRunoff"]   # runoff: rainfall-runoff model
            self.routing = model["Routing"] # routing: Routing
            self.abm = model.get("ABM")     # abm can be none (decoupled model)
            self.sys_parsed_data = model["SystemParsedData"]
        except:
            logger.error("Model file is incomplete or error.")

        path = self.path
        ws = self.ws
        abm = self.abm
        self.start_date = to_datetime(ws["StartDate"], format="%Y/%m/%d")
        self.data_length = ws["DataLength"]
        start_date = self.start_date
        data_length = self.data_length

        # Assign rainfall-runoff function
        ## Note the routing function is fixed to Lohmann.
        if ws["RainfallRunoff"] == "GWLF":
            self.runoff_func = run_GWLF
            logger.info("Set rainfall-runoff to GWLF.")
        elif ws["RainfallRunoff"] == "ABCD":
            self.runoff_func = run_ABCD
            logger.info("Set rainfall-runoff to ABCD.")
        else:
            logger.info("No assigned rainfall-runoff model.")

        # Initialize data_collector
        self.dc = Data_collector()  # For collecting ABM's data.
        dc = self.dc
        dc.add_field("Q_routed", {},
                     desc="Routed streamflow of routing outlets.",
                     unit="cms")
        dc.add_field("Q_runoff", {},
                     desc="Subbasin runoffs of outlets.",
                     unit="cms")
        dc.add_field("prec", {}, desc="Precipitation.", unit="cm")
        dc.add_field("temp", {}, desc="Temperature.", unit="degC")
        dc.add_field("pet", {}, desc="Potential evapotranspiration.",
                     unit="cm")
        dc.add_field("UH_Lohmann", {},
                     desc="Unit hydrograph of Lohmann routing.")
        dc.add_field("UH_Lohmann_convey", {},
                     desc="Unit hydrograph of Lohmann routing for convey agent.")

        # ----- Load external modules -----------------------------------------
        # We will automatically detect whether the ABM section is available.
        # If ABM section is not found, then we consider it as none coupled
        # model.
        # Technical note:
        # We will load user-defined modules (e.g., AgentType.py) into
        # HydroCNHS and store them under the UserModules class. Then, User
        # object is created for HydroCNHS to apply those user-defined classes.
        # We use eval() to turn string into python variable.
        # Please check the documentation for detailed instructions for
        # designing proper modules for HydroCNHS,. Certain protocals have to be
        # followed.


        self.agents = {}     # Store all agent objects {agt_id: agt object}.
        self.dms = {}        # Store all dm objects {agt_id: dm object}.
        self.instit_dms = {} # Store all institutional dm objects
                             # {institution: dm object}.
        agents = self.agents
        dms = self.dms
        instit_dms = self.instit_dms

        if ws["ABM"] is not None:
            # Import user-defined module --------------------------------------
            module_path = path.get("Modules")
            if module_path is not None:
                # User class will store all user-defined modules' classes and
                # functions.
                class UserModules:
                    pass
                for module_name in ws["ABM"]["Modules"]:
                    load_customized_module_to_class(UserModules, module_name,
                                                    module_path)
            user_object_name_list = list_callable_public_object(UserModules)

            # Initialize agents and decision-making objects ---------------
            for ag_type, agt in abm.items():
                for agt_id, ag_config in agt.items():
                    # Initialize agent objects
                    if ag_type in user_object_name_list:
                        # Load from user-defined module.
                        try:
                            agents[agt_id] = eval("UserModules."+ag_type)(
                                name=agt_id, config=ag_config, start_date=start_date,
                                current_date=start_date, data_length=data_length,
                                t=0, dc=dc, rn_gen=rn_gen)
                            logger.info(
                                "Create {} from {} class".format(agt_id, ag_type))
                        except Exception as e:
                            logger.error(traceback.format_exc())
                            raise Error(
                                "Fail to create {} from {} class.".format(agt_id, ag_type)
                                +"\nMake sure the class is well-defined in "
                                +"given modules.") from e
                    else:
                        # Try to load from built-in classes.
                        try:
                            agents[agt_id] = eval(ag_type)(
                                name=agt_id, config=ag_config, start_date=start_date,
                                current_date=start_date, data_length=data_length,
                                t=0, dc=dc, rn_gen=rn_gen)
                            logger.info(
                                "Create {} from {} ".format(agt_id, ag_type)
                                +"from the built-in classes.")
                        except Exception as e:
                            logger.error(traceback.format_exc())
                            raise Error(
                                "Fail to create {} from {} class.".format(agt_id, ag_type)
                                +"\n{} is not a built-in class.".format(ag_type)
                                ) from e

                    # Initialize dm or instit_dm is given in an agent object.
                    dm_name = abm[ag_type][agt_id]["Inputs"].get("DMClass")
                    ## instit_dm
                    if dm_name is not None:
                        instit_list = list(ws["ABM"]["Institutions"].keys())
                        if (dm_name in instit_list
                            and dm_name in list(instit_dms.keys())):
                            # Add institutional decision-making object to the agent object.
                            # Agents belong to a institute will share one dm object.
                            agents[agt_id].dm = instit_dms[dm_name]
                        elif (dm_name in instit_list
                            and dm_name not in list(instit_dms.keys())):
                            d = ws["ABM"]["InstitutionalDM"]
                            instit_dm_class = list(d.keys())[[dm_name in v \
                                for v in list(d.values())].index(True)]
                            try:
                                instit_dms[dm_name] = eval(
                                    "UserModules."+instit_dm_class)(
                                        name=dm_name, dc=dc, rn_gen=rn_gen)
                                logger.info(
                                    "Create institute {} from {} class.".format(
                                    dm_name, instit_dm_class))
                            except Exception as e:
                                logger.error(traceback.format_exc())
                                raise Error(
                                    "Fail to create institute {} from {}".format(
                                    dm_name, instit_dm_class)+" class.") from e
                            agents[agt_id].dm = instit_dms[dm_name]
                        # No built-in for institution
                        if (dm_name in user_object_name_list
                            and dm_name not in instit_list):
                            try:
                                dms[agt_id] = eval(
                                    "UserModules."+dm_name)(
                                        name=agt_id, dc=dc, rn_gen=rn_gen)
                                logger.info(
                                    "Create {} from {} class.".format(
                                    dm_name, dm_name))
                            except Exception as e:
                                logger.error(traceback.format_exc())
                                raise Error(
                                    "Fail to create {} from {} class.".format(
                                    dm_name, dm_name)) from e
                            agents[agt_id].dm = dms[agt_id]
                        elif (dm_name not in user_object_name_list
                            and dm_name not in instit_list):
                            try:
                                dms[agt_id] = eval(dm_name)(
                                        name=agt_id, dc=dc, rn_gen=rn_gen)
                                logger.info(
                                    "Create {} from built-in class.".format(
                                    dm_name, dm_name))
                            except Exception as e:
                                logger.error(traceback.format_exc())
                                raise Error(
                                    "Fail to create {} from built-in class.".format(
                                    dm_name, dm_name)) from e
                            agents[agt_id].dm = dms[agt_id]

            # Make all agents accessible to all agents
            for name, agt in agents.items():
                agt.agents = agents

    def load_weather_data(self, temp, prec, pet=None, outlets=[]):
        """Load temperature, precipitation, and otential evapotranpiration data.

        Parameters
        ----------
        temp : dict
            [degC] Daily mean temperature time series data (value) for each
            subbasin named by its outlet. E.g., {"subbasin1":[...],
            "subbasin2":[...]}
        prec : dict
            [cm] Daily precipitation time series data (value) for each
            subbasin named by its outlet. E.g., {"subbasin1":[...],
            "subbasin2":[...]}
        pet : dict, optional
            [cm] Daily potential evapotranpiration time series data (value) for
            each subbasin named by its outlet, by default None. E.g.,
            {"subbasin1":[...], "subbasin2":[...]}
        """
        ws = self.ws
        runoff = self.runoff
        logger = self.logger
        if pet is None:
            pet = {}
            # Default: calculate pet with Hamon's method and no dz adjustment.
            for sb in outlets:
                pet[sb] = cal_pet_Hamon(temp[sb],
                                        runoff[sb]["Inputs"]["Latitude"],
                                        ws["StartDate"], dz=None)
            logger.info("Compute pet by Hamon method. Users can improve "
                            +"the efficiency by assigning pre-calculated pet.")
        self.dc.temp = temp
        self.dc.prec = prec
        self.dc.pet = pet
        #self.weather = {"temp":temp, "prec":prec, "pet":pet}
        logger.info("Load temp & prec & pet with total length "
                         +"{}.".format(ws["DataLength"]))

    def run(self, temp, prec, pet=None, assigned_Q={}, assigned_UH={},
                 disable=False):
        """Run HydroCNHS simulation.

        Parameters
        ----------
        temp : dict
            [degC] Daily mean temperature.
        prec : dict
            [cm] Daily precipitation.
        pet : dict, optional
            [cm] Potential evapotranspiration, by
            default None. If none, pet is calculted by Hamon's method.
        assigned_Q : dict, optional
            [cms] If user want to manually assign Q for certain outlets
            {"outlet": array}, by default {}.
        assigned_UH : dict, optional
            If user want to manually assign UH (Lohmann) for certain outlet
            {"outlet": array}, by default {}.
        disable : bool, optional
            Disable display progress bar, by default False.

        Returns
        -------
        dict
            A dictionary of flow time series.
        """
        # Variables
        paral_setting = self.paral_setting
        sys_parsed_data = self.sys_parsed_data
        routing_outlets = sys_parsed_data["RoutingOutlets"]
        ws = self.ws
        start_date = self.start_date
        data_length = self.data_length
        runoff = self.runoff
        routing = self.routing
        name = self.name
        logger = self.logger
        # Data collector/container.
        # This dc will be passed around HydroCNHS and ABM.
        dc = self.dc

        # ----- Start a timer -------------------------------------------------
        start_time = time.monotonic()
        self.elapsed_time = 0
        def get_elapsed_time():
            elapsed_time = time.monotonic() - start_time
            self.elapsed_time = time.strftime("%H:%M:%S",
                                              time.gmtime(elapsed_time))
            return self.elapsed_time

        # ----- Rainfall-runoff simulation ------------------------------------
        Q_runoff = dc.Q_runoff
        outlets = ws["Outlets"]
        # Remove sub-basin that don't need to be simulated.
        outlets = list(set(outlets) - set(assigned_Q.keys()))
        # Load weather (and calculate pet with Hamon's method).
        self.load_weather_data(temp, prec, pet, outlets)
        # weather = self.weather
        # Update routing setting. No in-grid routing.
        if assigned_Q != {}:
            for ro in routing_outlets:
                for sb in routing[ro]:
                    if sb in assigned_Q:
                        # No in-grid routing.
                        routing[ro][sb]["Pars"]["GShape"] = None
                        routing[ro][sb]["Pars"]["GScale"] = None
                        logger.info(
                            "Turn {}'s GShape and GScale to ".format((sb, ro))
                            +"None in the routing setting. There is no "
                            +"in-grid time lag with given observed Q.")

        # Start runoff simulation in parallel.
        logger.info("Pre-calculate rainfall-runoffs for {} sub-basins. [{}]".format(
            len(outlets), get_elapsed_time()))
        QParel = Parallel(n_jobs=paral_setting["cores_runoff"],
                            verbose=paral_setting["verbose"]) \
                        ( delayed(self.runoff_func)\
                            (pars=runoff[sb]["Pars"], inputs=runoff[sb]["Inputs"],
                                temp=dc.temp[sb], prec=dc.prec[sb],
                                pet=dc.pet[sb], start_date=ws["StartDate"],
                                data_length=data_length) \
                            for sb in outlets )
        # ----- Add user assigned Q first. ------------------------------------
        # Collect QParel results
        Q_runoff = deepcopy(assigned_Q)    # Necessary deepcopy!
        for i, sb in enumerate(outlets):
            Q_runoff[sb] = QParel[i]

        # Q_routed will be continuously updated for routing.
        # Necessary deepcopy to isolate self.Q_runoff and self.Q_routed storage
        # pointer!
        dc.Q_routed = deepcopy(Q_runoff)
        logger.info("Complete rainfall-runoff simulation. [{}]".format(
            get_elapsed_time()))

        # ----- Form UH for Lohmann routing method ----------------------------
        # if ws_abm["Routing"] == "Lohmann": # No other choice
        # Form combination
        UH_List = [(sb, ro) for ro in routing_outlets \
                    for sb in routing[ro]]
        # Remove assigned UH from the list.
        UH_List_Lohmann = list(set(UH_List) - set(assigned_UH.keys()))
        # Start forming UH_Lohmann in parallel.
        logger.info(
            "Start forming {} UHs for Lohmann routing. [{}]".format(
                len(UH_List_Lohmann), get_elapsed_time()))
        # pair = (outlet, routing outlet)
        UHParel = Parallel(n_jobs=paral_setting["cores_formUH"],
                            verbose=paral_setting["verbose"]) \
                        ( delayed(form_UH_Lohmann)\
                        (routing[pair[1]][pair[0]]["Inputs"],
                            routing[pair[1]][pair[0]]["Pars"]) \
                        for pair in UH_List_Lohmann )

        # Form UH ---------------------------------------------------------
        # Add user assigned UH first.
        UH_Lohmann = dc.UH_Lohmann
        UH_Lohmann = deepcopy(assigned_UH)  # Necessary deepcopy!
        for i, pair in enumerate(UH_List_Lohmann):
            UH_Lohmann[pair] = UHParel[i]
        logger.info(
            "Complete forming UHs for Lohmann routing. [{}]".format(
                get_elapsed_time()))

        # Form UH for conveyed nodes --------------------------------------
        # No in-grid routing.
        conveyed_nodes = sys_parsed_data["ConveyToNodes"]
        UH_Lohmann_convey = dc.UH_Lohmann_convey
        if conveyed_nodes != []:
            UH_convey_List = []
            for uh in UH_List:
                if uh[0] in conveyed_nodes:
                    if uh in list(assigned_UH.keys()):
                        logger.error(
                            "Cannot process routing of conveying agents "
                            +"since {} unit hydrograph is assigned. We will "
                            +"use the assigned UH for simulation; however, "
                            +"the results might not be accurate.".format(uh))
                        UH_Lohmann_convey = UH_Lohmann[uh]
                    else:
                        UH_convey_List.append(uh)
            UHParel = Parallel(n_jobs=paral_setting["cores_formUH"],
                            verbose=paral_setting["verbose"]) \
                            ( delayed(form_UH_Lohmann)\
                            (routing[pair[1]][pair[0]]["Inputs"],
                            routing[pair[1]][pair[0]]["Pars"],
                            force_ingrid_off=True) \
                            for pair in UH_convey_List )
            for i, pair in enumerate(UH_convey_List):
                UH_Lohmann_convey[pair] = UHParel[i]
            logger.info(
                "Complete forming UHs for conveyed nodes. [{}]".format(
                    get_elapsed_time()))

        # ----- Time step simulation (Coupling hydrological model and ABM) ----
        Q_routed = dc.Q_routed
        # Obtain datetime index -----------------------------------------------
        pd_date_index = date_range(start=start_date, periods=data_length,
                                   freq="D")
        self.pd_date_index = pd_date_index  # So users can use it directly.

        # Load system-parsed data ---------------------------------------------
        sim_seq = sys_parsed_data["SimSeq"]
        ag_sim_seq = sys_parsed_data["AgSimSeq"]
        instream_agents = sys_parsed_data["DamAgents"]
        if instream_agents is None:
            instream_agents = []

        # Add instream agent to Q_routed --------------------------------------
        # instream_agents include DamAgents
        for isag in instream_agents:
            Q_routed[isag] = np.zeros(data_length)

        ##### Only a semi-distributed hydrological model ######################
        #####                     (Only RainfallRunoff and Routing)
        if ws["ABM"] is None:
            logger.info("Start a pure hydrological simulation (no human component).")
            # Run step-wise routing to update Q_routed ------------------------
            for t in tqdm(range(data_length), desc=name, disable=disable):
                current_date = pd_date_index[t]
                for node in sim_seq:
                    if node in routing_outlets:
                        #----- Run Lohmann routing model for one routing outlet
                        # (node) for 1 timestep (day).
                        Qt = run_step_Lohmann(node, routing, UH_Lohmann,
                                                Q_routed, Q_runoff, t)
                        #----- Store Qt to final output.
                        Q_routed[node][t] = Qt

        ##### HydroCNHS model (Coupled model) #################################
        # We create four interfaces for "two-way coupling" between natural
        # model and human model. However, the user-defined human model has to
        # follow specific protocal. See the documantation for details.
        else:
            logger.info("Start a HydroCNHS simulation.")
            agents = self.agents
            ### Add the storage for convey water.
            Q_convey = {}
            for c_node in conveyed_nodes:
                Q_convey[c_node] = np.zeros(data_length)

            for t in tqdm(range(data_length), desc=name, disable=disable):
                current_date = pd_date_index[t]
                # Update agents' attributes to new timestep.
                for name, agt in agents.items():
                    agt.current_date = current_date
                    agt.t = t

                for node in sim_seq:
                    # Load active agent for current node at time t ------------
                    river_div_ags_plus = ag_sim_seq["AgSimPlus"][node].get(
                        "RiverDivAgents")
                    river_div_ags_minus = ag_sim_seq["AgSimMinus"][node].get(
                        "RiverDivAgents")
                    insitu_ags_minus = ag_sim_seq["AgSimMinus"][node].get(
                        "InsituAgents")
                    insitu_ags_plus = ag_sim_seq["AgSimPlus"][node].get(
                        "InsituAgents")
                    dam_ags_plus = ag_sim_seq["AgSimPlus"][node].get(
                        "DamAgents")
                    convey_ags_plus = ag_sim_seq["AgSimPlus"][node].get(
                        "ConveyingAgents")
                    convey_ags_minus = ag_sim_seq["AgSimMinus"][node].get(
                        "ConveyingAgents")

                    # Note for the first three if, we should only enter one of
                    # them at each node.
                    if (insitu_ags_minus is not None
                        or insitu_ags_plus is not None
                        or river_div_ags_plus is not None):
                        r"""
                        For InsituAPI, they change water directly from
                        the runoff in each sub-basin or grid.
                        Note that InsituAPI has no return flow option.
                        After updating Q generated by RainfallRunoff and plus
                        the return flow, we run the routing to calculate the
                        routing streamflow at the routing outlet stored in
                        Q_routed.
                        Note that we need to use the updated self.Q_runoff =
                                self.Q_runoff - Div + return flow
                        for the routing outlet to run the routing model.
                        That means return flow will join the in-grid routing!
                        Therefore, in this section, both self.Q_routed and
                        self.Q_runoff will be updated.
                        """
                        if insitu_ags_plus is not None:
                            for ag, o in insitu_ags_plus:
                                delta = agents[ag].act(outlet=o)
                                Q_routed[o][t] += delta
                                # For routing outlet within-subbasin routing.
                                Q_runoff[o][t] += delta

                        if insitu_ags_minus is not None:
                            for ag, o in insitu_ags_minus:
                                delta = agents[ag].act(outlet=o)
                                Q_routed[o][t] += delta
                                # For routing outlet within-subbasin routing.
                                Q_runoff[o][t] += delta

                        if river_div_ags_plus is not None:
                            # e.g., add return flow
                            for ag, o in river_div_ags_plus:
                                # self.Q_runoff + return flow
                                # return flow will join the within-subbasin routing.
                                delta = agents[ag].act(outlet=o)
                                # For returning to other outlets.
                                Q_routed[o][t] += delta
                                # For routing outlet within-subbasin routing.
                                Q_runoff[o][t] += delta

                    elif dam_ags_plus is not None:
                        r"""
                        For DamAgents, we simply add the release water to
                        self.Q_routed[isag]. No minus action is needed for its
                        upstream inflow outlet.
                        """
                        for ag, o in dam_ags_plus:
                            delta = agents[ag].act(outlet=o)
                            Q_routed[o][t] += delta

                    if convey_ags_plus is not None:
                        # Don't have in-grid routing.
                        for ag, o in convey_ags_plus:
                            delta = agents[ag].act(outlet=o)
                            Q_convey[o][t] += delta

                    if node in routing_outlets:
                        #----- Run Lohmann routing model for one routing outlet
                        # (node) for 1 time step (day).
                        Qt = run_step_Lohmann(
                            node, routing, UH_Lohmann, Q_routed, Q_runoff, t)
                        Qt_convey = run_step_Lohmann_convey(
                            node, routing, UH_Lohmann_convey, Q_convey, t)
                        #----- Store Qt to final output.
                        Q_routed[node][t] = Qt + Qt_convey

                    if convey_ags_minus is not None:
                        for ag, o in convey_ags_minus:
                            delta = agents[ag].act(outlet=o)
                            Q_routed[o][t] += delta

                    if river_div_ags_minus is not None:
                        r"""
                        For river_div_ags_minus, we divert water from the
                        routed river flow at agent-associated routing outlet.
                        """
                        for ag, o in river_div_ags_minus:
                            delta = agents[ag].act(outlet=o)
                            Q_routed[o][t] += delta
        # ---------------------------------------------------------------------
        print("")   # Force the logger to start a new line after tqdm.
        logger.info(
            "Complete HydroCNHS simulation! [{}]\n".format(get_elapsed_time()))
        # [cms] Streamflow for routing outlets (Gauged outlets and inflow
        # outlets of instream agents). For other variables users need to
        # extract them manually from this class.
        return Q_routed

    def get_model_object(self):
        """Get the model object in a dictionary form.

        Returns
        -------
        dict
            model object dictionary.
        """
        return self.__dict__