# Main HydroCNHS simulator.
# This module is design to couple the human model with semi-distributed 
# hydrological model to form a coupled natural human system.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# 2021/02/05

import time
import traceback
from copy import Error, deepcopy    # For deepcopy dictionary.
import numpy as np
from pandas import date_range, to_datetime
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
from .land_surface_model.abcd import run_ABCD
from .land_surface_model.gwlf import run_GWLF
from .land_surface_model.hymod import run_HYMOD
from .land_surface_model.pet_hamon import cal_pet_Hamon
from .routing import form_UH_Lohmann, run_step_Lohmann, run_step_Lohmann_convey
from .util import (load_system_config, load_model,
                   load_customized_module_to_class)
from .data_collector import Data_collector

class Model(object):
    def __init__(self, model, name=None, rn_gen=None, checked=False, parsed=False):
        """HydroCNHS model object.

        Args:
            model (str/dict): model.yaml file (prefer) or dictionary. 
            name ([str], optional): Object name. Defaults to None.
        """
        # Assign model name and get logger.
        self.name = name
        if name is None:
            self.logger = logging.getLogger("HydroCNHS") # Get logger 
        else:
            self.logger = logging.getLogger("HydroCNHS."+name) # Get logger 
        
        # Load HydroCNHS system configuration.
        self.sys_config = load_system_config()   
            
        # Load model.yaml and distribute into several variables.
        ## We design model to be either str or dictionary.
        model = load_model(model, checked=checked, parsed=parsed)
        
        # Create random number generator for feeding into ABM.
        if rn_gen is None:
            # Assign a random seed.
            seed = np.random.randint(0, 100000)
            self.rn_gen = np.random.default_rng(seed)
        else:
            # User-provided rn generator
            self.rn_gen = rn_gen
            self.ss = rn_gen.bit_generator._seed_seq
            self.logger.info(
                "User-provided random number generator is assigned.")
            
        # Verify model contain all following keys.
        try:                   
            self.path = model["Path"]
            self.ws = model["WaterSystem"]  # ws: Water system
            self.lsm = model["LSM"]         # lsm: Land surface model
            self.routing = model["Routing"] # routing: Routing 
            self.abm = model.get("ABM")     # abm can be none (decoupled model)
            self.sys_parsed_data = model["SystemParsedData"]
        except:
            self.logger.error("Model file is incomplete for HydroCNHS.")
            
        # Initialize output
        self.Q_routed = {}     # [cms] Streamflow for routing outlets.
        self.data_collector = Data_collector()  # For collecting ABM's data.

    def load_weather_data(self, temp, prec, pet=None, lsm_outlets=None):
        """[Include in run] Load temperature and precipitation data.
        Can add some check functions or deal with miss values here.
        Args:
            temp (dict): [degC] Daily mean temperature time series data (value)
                for each sub-basin named by its outlet.
            prec (dict): [cm] Daily precipitation time series data (value) for
                each sub-basin named by its outlet.
            pet(dict/None): [cm] Daily potential evapotranpiration time series
                data (value) for each sub-basin named by its outlet.
            LSM_outlets(dict/None): Should equal to self.ws["Outlets"]
        """
        ws = self.ws
        lsm = self.lsm
        logger = self.logger
        if pet is None:
            pet = {}
            # Default: calculate pet with Hamon's method and no dz adjustment.
            for sb in lsm_outlets:
                pet[sb] = cal_pet_Hamon(temp[sb],
                                        lsm[sb]["Inputs"]["Latitude"],
                                        ws["StartDate"], dz=None)
            logger.info("Compute pet by Hamon method. Users can improve "
                            +"the efficiency by assigning pre-calculated pet.")
        self.weather = {"temp":temp, "prec":prec, "pet":pet}
        logger.info("Load temp & prec & pet with total length "
                         +"{}.".format(ws["DataLength"]))
           
    def run(self, temp, prec, pet=None, assigned_Q={}, assigned_UH={},
                 disable=False):
        """Run HydroCNHS simulation.
        
        Args:
            temp (dict): [degC] Daily mean temperature.
            prec (dict): [cm] Daily precipitation.
            pet (dict, optional): [cm] Potential evapotranspiration calculted
                by Hamon's method. Defaults to None.
            assigned_Q (dict, optional): [cms] If user want to manually assign
                Q for certain outlet {"outlet": array}. Defaults to None.
            assigned_UH (dict, optional): If user want to manually assign UH
                (Lohmann) for certain outlet {"outlet": array}. Defaults to
                None.
            disable (bool): Disable tqdm. Defaults to False.
        """
        # Variables
        paral_setting = self.sys_config["Parallelization"]
        sys_parsed_data = self.sys_parsed_data
        routing_outlets = sys_parsed_data["RoutingOutlets"]
        ws = self.ws
        start_date = to_datetime(ws["StartDate"], format="%Y/%m/%d")  
        data_length = ws["DataLength"]
        lsm = self.lsm
        routing = self.routing
        abm = self.abm
        path = self.path
        name = self.name
        logger = self.logger
        rn_gen = self.rn_gen
        
        # ----- Start a timer -------------------------------------------------
        start_time = time.monotonic()
        self.elapsed_time = 0
        def get_elapsed_time():
            elapsed_time = time.monotonic() - start_time
            self.elapsed_time = time.strftime("%H:%M:%S",
                                              time.gmtime(elapsed_time))
            return self.elapsed_time
        
        # ----- Load external modules -----------------------------------------
        # We will automatically detect whether the ABM section is available. 
        # If ABM section is not found, then we consider it as none coupled 
        # model.
        # Technical note:
        #   We will load user-defined modules (e.g., AgentType.py) into 
        # HydroCNHS and store them under the UserModules class. Then, User 
        # object is created for Hydro CNHS to apply those user-defined classes.
        # We use eval() to turn string into python variable.
        #   Detailed instruction for designing proper modules for HydroCNHS, 
        # please check the documentation. Certain protocals have to be followed.
        
        self.agents = {}     # Store all agent objects with key = agentname.
        self.DM_classes = {}   # Store all DM function Ex {"DMFunc": DMFunc()}
        agents = self.agents
        DM_classes = self.DM_classes
        
        if abm is not None: 
            # Data collector
            dc = self.data_collector
            
            # Import user-defined module --------------------------------------
            module_path = path.get("Modules")
            if module_path is not None:
                # User class will store all user-defined modules' classes and 
                # functions.
                class UserModules:
                    pass
                for module_name in abm["Inputs"]["Modules"]:
                    load_customized_module_to_class(UserModules, module_name,
                                                    module_path)
            
            # Collect user-defined DMClasses ----------------------------------
            for dmclass in abm["Inputs"]["DMClasses"]:
                try:    # Try to load from user-defined module first.
                    DM_classes[dmclass] = eval("UserModules."+dmclass)
                    logger.info(
                        "Load {} from the user-defined classes.".format(
                            dmclass))
                except Exception as e:
                    try:    # Detect if it is a built-in class.
                        DM_classes[dmclass] = eval(dmclass)
                        logger.info(
                            "Load {} from the built-in classes.".format(
                                dmclass))
                    except Exception as _:
                        logger.error(traceback.format_exc())
                        logger.debug(e)
                        raise Error("Fail to load {}.\n".format(dmclass)
                                    +"Make sure the class is well-defined in "
                                    +"given modules.")
            
            r"""
            Temp save
                        for dmclass in abm["Inputs"]["DMClasses"]:
                try:    # Try to load from user-defined module first.
                    DM_classes[dmclass] = eval("UserModules."+dmclass)(
                        start_date, data_length, abm)
                    logger.info(
                        "Load {} from the user-defined classes.".format(
                            dmclass))
                except Exception as e:
                    try:    # Detect if it is a built-in class.
                        DM_classes[dmclass] = eval(dmclass)(
                            start_date, data_length, abm)
                        logger.info(
                            "Load {} from the built-in classes.".format(
                                dmclass))
                    except Exception as _:
                        logger.error(traceback.format_exc())
                        logger.debug(e)
                        raise Error("Fail to load {}.\n".format(dmclass)
                                    +"Make sure the class is well-defined in "
                                    +"given modules.")
            """
            
            # Initialize agent action groups ----------------------------------
            # The agent action groups is different from the DMFunc. Action 
            # group do actions (e.g., divert water) together based on their 
            # original decisions (e.g., diversion request) from DMFunc. 
            # This could be used in a situation, where agents share the water 
            # deficiency together. 
            # AgGroup = {"AgType":{"Name": []}}   (in Model.yaml)
            ag_group = abm["Inputs"].get("AgGroup")
            if ag_group is not None:
                for ag_type in ag_group:
                    for agG in ag_group[ag_type]:
                        agList = ag_group[ag_type][agG]
                        ag_config = {}
                        for ag in agList:
                            ag_config[ag] = abm[ag_type][ag]
                        try:      # Try to load from user-defined module first.
                            agents[agG] = eval("UserModules."+ag_type)(
                                name=agG, config=ag_config,
                                start_date=start_date, data_length=data_length,
                                data_collector=dc, rn_gen=rn_gen)
                            # Add dm class to Agent.
                            dm_name = abm[ag_type]["Inputs"].get("DMClass")
                            dm_class = DM_classes.get(dm_name)
                            if dm_class is None:
                                agents[agG].dm = None
                                logger.info(
                                    "DMClass of {}".format(ag_type)
                                    +"is not defined.")
                            else:
                                agents[agG].dm = dm_class(
                                    start_date=start_date,
                                    data_length=data_length, abm=abm,
                                    data_collector=dc, rn_gen=rn_gen)
                            logger.info(
                                "Load {} for {} ".format(ag_type, agG)
                                +"from the user-defined classes.")
                        except Exception as e:
                            try:  # Detect if it is a built-in class.
                                logger.info(
                                    "Try to load {} for {} ".format(ag_type,
                                                                    agG)
                                    +"from the built-in classes.")
                                agents[agG] = eval(ag_type)(
                                    name=agG, config=ag_config,
                                    start_date=start_date,
                                    data_length=data_length,
                                    data_collector=dc, rn_gen=rn_gen)
                                logger.info(
                                    "Load {} for {} ".format(ag_type, agG)
                                    +"from the built-in classes.")
                            except Exception as e:
                                logger.debug(e)
                                logger.error(traceback.format_exc())
                                raise Error(
                                    "Fail to load {} for {}.".format(ag_type,
                                                                     agG)
                                    +"\nMake sure the class is well-defined "
                                    +"in given modules.")
            else:
                ag_group = []
                
            # Initialize agents not belong to any action groups ---------------
            for ag_type, Ags in abm.items():
                if ag_type == "Inputs" or ag_type in ag_group:
                        continue
                for ag, ag_config in Ags.items():
                    try:        # Try to load from user-defined module first.
                        agents[ag] = eval("UserModules."+ag_type)(
                            name=ag, config=ag_config, start_date=start_date,
                            data_length=data_length, data_collector=dc,
                            rn_gen=rn_gen)
                        # Add dm class to Agent.
                        dm_name = abm[ag_type][ag]["Inputs"].get("DMClass")
                        dm_class = DM_classes.get(dm_name)
                        if dm_class is None:
                            agents[ag].dm = None
                            logger.info(
                                "DMClass of {}".format(ag)
                                +"is not defined.")
                        else:
                            agents[ag].dm = dm_class(
                                start_date=start_date,
                                data_length=data_length, abm=abm,
                                data_collector=dc, rn_gen=rn_gen)
                        logger.info(
                            "Load {} for {} ".format(ag_type, ag)
                            +"from the user-defined classes.")
                    except Exception as e:
                        try:    # Detect if it is a built-in class.
                            agents[ag] = eval(ag_type)(
                                name=ag, config=ag_config,
                                start_date=start_date, data_length=data_length,
                                data_collector=dc, rn_gen=rn_gen)
                            logger.info(
                                "Load {} for {} ".format(ag_type, ag)
                                +"from the built-in classes.")
                        except Exception as e:
                            logger.error(traceback.format_exc())
                            raise Error(
                                "Fail to load {} for {}.".format(ag_type, ag)
                                +"\nMake sure the class is well-defined in "
                                +"given modules.")
        
        # ----- Land surface simulation ---------------------------------------
        self.Q_LSM = {}
        outlets = ws["Outlets"]
        # Remove sub-basin that don't need to be simulated. 
        outlets = list(set(outlets) - set(assigned_Q.keys()))  
        # Load weather (and calculate pet with Hamon's method).
        self.load_weather_data(temp, prec, pet, outlets) 
        weather = self.weather
        # Update routing setting. No in-grid routing.
        if assigned_Q != {}:
            for ro in routing_outlets:
                for sb in routing[ro]:
                    if sb in assigned_Q:
                        # No in-grid routing.
                        routing[ro][sb]["Pars"]["GShape"] = None  
                        routing[ro][sb]["Pars"]["GRate"] = None   
                        logger.info(
                            "Turn {}'s GShape and GRate to ".format((sb, ro))
                            +"None in the routing setting. There is no "
                            +"in-grid time lag with given observed Q.")
        
        # Start GWLF simulation in parallel.
        if lsm["Model"] == "GWLF":
            logger.info("Start GWLF for {} sub-basins. [{}]".format(
                len(outlets), get_elapsed_time()))
            QParel = Parallel(n_jobs=paral_setting["Cores_LSM"],
                              verbose=paral_setting["verbose"]) \
                            ( delayed(run_GWLF)\
                                (lsm[sb]["Pars"], lsm[sb]["Inputs"],
                                 weather["temp"][sb], weather["prec"][sb],
                                 weather["pet"][sb], ws["StartDate"],
                                 data_length) \
                                for sb in outlets ) 
                            
        # Start HYMOD simulation in parallel.
        # Not verify this model yet.
        if lsm["Model"] == "HYMOD":
            logger.info("Start HYMOD for {} sub-basins. [{}]".format(
                len(outlets), get_elapsed_time()))   
            QParel = Parallel(n_jobs=paral_setting["Cores_LSM"],
                              verbose=paral_setting["verbose"]) \
                            ( delayed(run_HYMOD)\
                                (lsm[sb]["Pars"], lsm[sb]["Inputs"],
                                 weather["temp"][sb], weather["prec"][sb],
                                 weather["pet"][sb], data_length) \
                                for sb in outlets ) 
        
        # Start ABCD simulation in parallel.
        # Not verify this model yet.
        if lsm["Model"] == "ABCD":
            logger.info("Start ABCD for {} sub-basins. [{}]".format(
                len(outlets), get_elapsed_time())) 
            QParel = Parallel(n_jobs=paral_setting["Cores_LSM"],
                              verbose=paral_setting["verbose"]) \
                            ( delayed(run_ABCD)\
                                (lsm[sb]["Pars"], lsm[sb]["Inputs"],
                                 weather["temp"][sb], weather["prec"][sb],
                                 weather["pet"][sb], data_length) \
                                for sb in outlets ) 

        # ----- Add user assigned Q first. ------------------------------------
        self.Q_LSM = deepcopy(assigned_Q)    # Necessary deepcopy!
        # Collect QParel results
        Q_LSM = self.Q_LSM
        for i, sb in enumerate(outlets):
            Q_LSM[sb] = QParel[i]
            
        # Q_routed will be continuously updated for routing.
        # Necessary deepcopy to isolate self.Q_LSM and self.Q_routed storage
        # pointer!
        self.Q_routed = deepcopy(Q_LSM)
        logger.info("Complete LSM simulation. [{}]".format(
            get_elapsed_time())) 
    
        # ----- Form UH for Lohmann routing method ----------------------------
        if routing["Model"] == "Lohmann":
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
            UHParel = Parallel(n_jobs=paral_setting["Cores_formUH_Lohmann"],
                               verbose=paral_setting["verbose"]) \
                            ( delayed(form_UH_Lohmann)\
                            (routing[pair[1]][pair[0]]["Inputs"],
                             routing[pair[1]][pair[0]]["Pars"]) \
                            for pair in UH_List_Lohmann )

            # Form UH ---------------------------------------------------------
            # Add user assigned UH first.
            self.UH_Lohmann = {}
            self.UH_Lohmann = deepcopy(assigned_UH)  # Necessary deepcopy!
            UH_Lohmann = self.UH_Lohmann
            for i, pair in enumerate(UH_List_Lohmann):
                UH_Lohmann[pair] = UHParel[i]
            logger.info(
                "Complete forming UHs for Lohmann routing. [{}]".format(
                    get_elapsed_time()))
            
            # Form UH for conveyed nodes --------------------------------------
            # No in-grid routing.
            conveyed_nodes = sys_parsed_data["ConveyToNodes"]
            self.UH_Lohmann_convey = {}
            UH_Lohmann_convey = self.UH_Lohmann_convey
            if conveyed_nodes != []:
                UH_convey_List = []
                for uh in UH_List:
                    if uh[0] in conveyed_nodes:
                        if uh in list(assigned_UH.keys()):
                            logger.error(
                                "Cannot process routing of conveying agents "
                                +"since {} unit hydrograph is assigned. We will "
                                +"use the assigned UH for simulation; however, "
                                +"the results might not be correct.".format(uh))
                            UH_Lohmann_convey = UH_Lohmann[uh]
                        else:
                            UH_convey_List.append(uh)
                UHParel = Parallel(n_jobs=paral_setting["Cores_formUH_Lohmann"],
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
        Q_routed = self.Q_routed
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
        #####                     (Only LSM and Routing)
        if abm is None: 
            logger.info("Start the non-coupled simulation.")
            # Run step-wise routing to update Q_routed ------------------------
            for t in tqdm(range(data_length), desc=name, disable=disable):
                current_date = pd_date_index[t]
                for node in sim_seq:
                    if node in routing_outlets:
                        #----- Run Lohmann routing model for one routing outlet
                        # (node) for 1 timestep (day).
                        if routing["Model"] == "Lohmann":
                            Qt = run_step_Lohmann(node, routing, UH_Lohmann,
                                                  Q_routed, Q_LSM, t)
                        #----- Store Qt to final output.
                        Q_routed[node][t] = Qt 
                        
        ##### HydroCNHS model (Coupled model) #################################
        # We create four interfaces for "two-way coupling" between natural  
        # model and human model. However, the user-defined human model has to 
        # follow specific protocal. See the documantation for details.
        else:
            logger.info("Start the HydroCNHS simulation.")
            
            ### Add the storage for convey water.
            Q_convey = {}
            for c_node in conveyed_nodes:
                Q_convey[c_node] = np.zeros(data_length)
            
            for t in tqdm(range(data_length), desc=name, disable=disable):
                current_date = pd_date_index[t]
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
                        "ConveyAgents")
                    convey_ags_minus = ag_sim_seq["AgSimMinus"][node].get(
                        "ConveyAgents")

                    # Note for the first three if, we should only enter one of 
                    # them at each node.
                    if (insitu_ags_minus is not None
                        or insitu_ags_plus is not None
                        or river_div_ags_plus is not None):
                        r"""
                        For InsituAgentTypes, they change water directly from 
                        the runoff in each sub-basin or grid.
                        Note that InsituAgentTypes has no return flow option.
                        After updating Q generated by LSM and plus the return 
                        flow, we run the routing to calculate the routing 
                        streamflow at the routing outlet stored in Q_routed.
                        Note that we need to use the updated self.Q_LSM = 
                                self.Q_LSM - Div + return flow
                        for the routing outlet to run the routing model.
                        That means return flow will join the in-grid routing!
                        Therefore, in this section, both self.Q_routed and 
                        self.Q_LSM will be updated.
                        """
                        if insitu_ags_plus is not None:
                            for ag, o in insitu_ags_plus:
                                delta = agents[ag].act(
                                    Q=Q_routed, outet=o, agent_dict=agents,
                                    current_date=current_date, t=t)
                                Q_routed[o][t] += delta
                                # For routing outlet in-grid routing.
                                Q_LSM[o][t] += delta
                                
                        if insitu_ags_minus is not None:
                            for ag, o in insitu_ags_minus:
                                delta = agents[ag].act(
                                    Q=Q_routed, outet=o, agent_dict=agents,
                                    current_date=current_date, t=t)
                                Q_routed[o][t] += delta
                                # For routing outlet in-grid routing.
                                Q_LSM[o][t] += delta
                                
                        if river_div_ags_plus is not None:    
                            # e.g., add return flow
                            for ag, o in river_div_ags_plus:
                                # self.Q_LSM + return flow   
                                # return flow will join the in-grid routing. 
                                delta = agents[ag].act(
                                    Q=Q_routed, outet=o, agent_dict=agents,
                                    current_date=current_date, t=t)
                                # For returning to other outlets.
                                Q_routed[o][t] += delta
                                # For routing outlet in-grid routing.
                                Q_LSM[o][t] += delta
                    
                    elif dam_ags_plus is not None:
                        r"""
                        For DamAgents, we simply add the release water to 
                        self.Q_routed[isag]. No minus action is needed for its 
                        upstream inflow outlet.
                        """
                        for ag, o in dam_ags_plus:
                            delta = agents[ag].act(
                                Q=Q_routed, outet=o, agent_dict=agents,
                                current_date=current_date, t=t)
                            Q_routed[o][t] += delta
                            
                    if convey_ags_plus is not None:    
                        # Don't have in-grid routing.
                        for ag, o in convey_ags_plus:
                            delta = agents[ag].act(
                                Q=Q_convey, outet=o, agent_dict=agents,
                                current_date=current_date, t=t)
                            Q_convey[o][t] += delta
                            
                    if node in routing_outlets:
                        #----- Run Lohmann routing model for one routing outlet
                        # (node) for 1 time step (day).
                        if routing["Model"] == "Lohmann":
                            Qt = run_step_Lohmann(
                                node, routing, UH_Lohmann, Q_routed, Q_LSM, t)
                            Qt_convey = run_step_Lohmann_convey(
                                node, routing, UH_Lohmann_convey, Q_convey, t)
                        #----- Store Qt to final output.
                        Q_routed[node][t] = Qt + Qt_convey
                    
                    if convey_ags_minus is not None:    
                        for ag, o in convey_ags_minus:
                            delta = agents[ag].act(
                                Q=Q_routed, outet=o, agent_dict=agents,
                                current_date=current_date, t=t)
                            Q_routed[o][t] += delta
                            
                    if river_div_ags_minus is not None:
                        r"""
                        For river_div_ags_minus, we divert water from the 
                        routed river flow at agent-associated routing outlet.
                        """
                        for ag, o in river_div_ags_minus:
                            delta = agents[ag].act(
                                Q=Q_routed, outet=o, agent_dict=agents,
                                current_date=current_date, t=t)
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
        return self.__dict__