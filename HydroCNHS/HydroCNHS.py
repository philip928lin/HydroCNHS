#%%
# Form the water system using a semi distribution hydrological model and agent-based model.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# 2021/02/05
from joblib import Parallel, delayed    # For parallelization
from pandas import date_range, to_datetime
from tqdm import tqdm
from copy import Error, deepcopy   # For deepcopy dictionary.
import traceback
import numpy as np
import time
import logging
logger = logging.getLogger("HydroCNHS") # Get logger 

from .LSM import runGWLF, calPEt_Hamon, runHYMOD
from .Routing import formUH_Lohmann, runTimeStep_Lohmann
from .SystemConrol import loadConfig, loadModel
from .Agent import *                    # AgType_Reservoir, AgType_IrrDiversion

class HydroCNHSModel(object):
    """Main HydroCNHS simulation object.

    """
    def __init__(self, model, name = None):
        """HydroCNHS constructor

        Args:
            model (str/dict): model.yaml file (prefer) or dictionary. 
            name ([str], optional): Object name. Defaults to None.
        """
        self.__name__ = name
        
        # Load HydroCNHS system configuration.
        self.Config = loadConfig()   
            
        # Load model.yaml and distribute into several variables.
        Model = loadModel(model)    # We design model to be str or dictionary.
        
        # Need to verify Model contain all following keys.
        
        try:                   
            self.Path = Model["Path"]
            self.WS = Model["WaterSystem"]     # WS: Water system
            self.LSM = Model["LSM"]            # LSM: Land surface model
            self.RR = Model["Routing"]         # RR: Routing 
            self.ABM = Model.get("ABM")        # ABM can be none (None coupled model)
            self.SysPD = Model["SystemParsedData"]
        except:
            logger.error("Model is incomplete for CNHS.")
            
        # Initialize output
        self.Q_routed = {}     # [cms] Streamflow for routing outlets (Gauged outlets and inflow outlets of in-stream agents).
        # self.A = {}     # Collect agent's output for each AgentType.
    
    def loadWeatherData(self, T, P, PE = None, LSMOutlets = None):
        """[Include in run] Load temperature and precipitation data.
        Can add some check functions or deal with miss values here.
        Args:
            T (dict): [degC] Daily mean temperature time series data (value) for each sub-basin named by its outlet.
            P (dict): [cm] Daily precipitation time series data (value) for each sub-basin named by its outlet.
            PE(dict/None): [cm] Daily potential evapotranpiration time series data (value) for each sub-basin named by its outlet.
            LSMOutlets(dict/None): Should equal to self.WS["Outlets"]
        """
        if PE is None:
            PE = {}
            # Default to calculate PE with Hamon's method and no dz adjustment.
            for sb in LSMOutlets:
                PE[sb] = calPEt_Hamon(T[sb], self.LSM[sb]["Inputs"]["Latitude"], self.WS["StartDate"], dz = None)
                logger.info("Compute PEt by Hamon method. Users can improve the efficiency by assigning pre-calculated PEt.")
        self.Weather = {"T":T, "P":P, "PE":PE}
        logger.info("Load T & P & PE with total length {}.".format(self.WS["DataLength"]))
           
    def __call__(self, T, P, PE = None, AssignedQ = {}, AssignedUH = {}, disable = False):
        """Run HydroCNHS simulation. The simulation is controled by model.yaml and Config.yaml (HydroCNHS system file).
        
        Args:
            T (dict): Daily mean temperature.
            P (dict): Daily precipitation.
            PE (dict, optional): Potential evapotranspiration. Defaults to None (calculted by Hamon's method).
            AssignedQ (dict, optional): If user want to manually assign Q (value, Array) for certain outlet (key, str). Defaults to None.
            AssignedUH (dict, optional): If user want to manually assign UH (Lohmann) (value, Array) for certain outlet (key, str). Defaults to None.
        """
        
        # Set a timer here
        start_time = time.monotonic()
        self.elapsed_time = 0
        def getElapsedTime():
            elapsed_time = time.monotonic() - start_time
            self.elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            return self.elapsed_time
        
        Paral = self.Config["Parallelization"]
        Outlets = self.WS["Outlets"]
        self.Q_LSM = {}  # Temporily store Q result from land surface simulation.
        
        # ----- Land surface simulation ---------------------------------------
        # Remove sub-basin that don't need to be simulated. Not preserving element order in the list.
        Outlets = list(set(Outlets) - set(AssignedQ.keys()))  
        if AssignedQ != {}:
            RoutingOutlets = self.SysPD["RoutingOutlets"]
            for ro in RoutingOutlets:
                for sb in self.RR[ro]:
                    if sb in AssignedQ:
                        self.RR[ro][sb]["Pars"]["GShape"] = None   # No in-grid routing.
                        self.RR[ro][sb]["Pars"]["GRate"] = None   # No in-grid routing.
                        logger.info("Turn {}'s GShape and GRate  to None in routing setting. Since Q (assuming to be observed data) is given, there is no in-grid time lag.".format((sb, ro)))
        
        # Start GWLF simulation in parallel.
        if self.LSM["Model"] == "GWLF":
            self.Q_GWLF = {}
            logger.info("[{}] Start GWLF for {} sub-basins. [{}]".format(self.__name__, len(Outlets), getElapsedTime()))
            # Load weather and calculate PEt with Hamon's method.
            self.loadWeatherData(T, P, PE, Outlets)    
            QParel = Parallel(n_jobs = Paral["Cores_runGWLF"], verbose = Paral["verbose"]) \
                            ( delayed(runGWLF)\
                                (self.LSM[sb]["Pars"], self.LSM[sb]["Inputs"], self.Weather["T"][sb], self.Weather["P"][sb], self.Weather["PE"][sb], self.WS["StartDate"], self.WS["DataLength"]) \
                                for sb in Outlets ) 
                            
        # Start HYMOD simulation in parallel.
        if self.LSM["Model"] == "HYMOD":
            logger.info("[{}] Start HYMOD for {} sub-basins. [{}]".format(self.__name__, len(Outlets), getElapsedTime()))
            # Load weather and calculate PEt with Hamon's method.
            self.loadWeatherData(T, P, PE, Outlets)    
            QParel = Parallel(n_jobs = Paral["Cores_runGWLF"], verbose = Paral["verbose"]) \
                            ( delayed(runHYMOD)\
                                (self.LSM[sb]["Pars"], self.LSM[sb]["Inputs"], self.Weather["P"][sb], self.Weather["T"][sb], self.Weather["PE"][sb], self.WS["DataLength"]) \
                                for sb in Outlets ) 

        # Add user assigned Q first.
        # Q_routed will be continuously updated for routing.
        self.Q_LSM = deepcopy(AssignedQ)            # Necessary deepcopy!
        self.Q_routed = deepcopy(AssignedQ)         # Necessary deepcopy!
        # Collect QParel results
        for i, sb in enumerate(Outlets):
            self.Q_LSM[sb] = QParel[i]
            self.Q_routed[sb] = QParel[i]
        logger.info("[{}] Complete GWLF... [{}]".format(self.__name__, getElapsedTime()))
        # ---------------------------------------------------------------------    
    

        # ----- Form UH for Lohmann routing method ----------------------------
        if self.RR["Model"] == "Lohmann":
            self.UH_Lohmann = {}    # Initialized the output dictionary
            RoutingOutlets = self.SysPD["RoutingOutlets"]
            # Form combination
            UH_List = [(sb, ro) for ro in RoutingOutlets for sb in self.RR[ro]]
            # Remove assigned UH from the list. Not preserving element order in the list.
            UH_List_Lohmann = list(set(UH_List) - set(AssignedUH.keys()))
            # Start forming UH_Lohmann in parallel.
            logger.info("[{}] Start forming {} UHs for Lohmann routing. [{}]".format(self.__name__, len(UH_List_Lohmann), getElapsedTime()))
            UHParel = Parallel(n_jobs = Paral["Cores_formUH_Lohmann"], verbose = Paral["verbose"]) \
                            ( delayed(formUH_Lohmann)\
                            (self.RR[pair[1]][pair[0]]["Inputs"], self.RR[pair[1]][pair[0]]["Pars"]) \
                            for pair in UH_List_Lohmann )     # pair = (Outlet, GaugedOutlet)
            # Add user assigned UH first.

            # Form UH
            self.UH_Lohmann = deepcopy(AssignedUH)  # Necessary deepcopy!
            for i, pair in enumerate(UH_List_Lohmann):
                self.UH_Lohmann[pair] = UHParel[i]
            logger.info("[{}] Complete forming UHs for Lohmann routing... [{}]".format(self.__name__, getElapsedTime()))
        # ---------------------------------------------------------------------

        
        # ----- Load Agents from ABM -----
        # Note: we will automatically detect whether the ABM section is available. If ABM section is not found,
        # then we consider it as none coupled model.
        StartDate = to_datetime(self.WS["StartDate"], format="%Y/%m/%d")  
        self.Agents = {}     # Here we store all agent objects with key = agent name.
        if self.ABM is not None: 
            for agType, Ags in self.ABM.items():
                if agType == "Inputs":
                        continue
                for ag, agConfig in Ags.items():
                    # eval(agType) will turn the string into class. Therefore, agType must be a well-defined class in Agent module.
                    try:
                        # Initialize agent object from agent-type class defined in Agent.py.
                        self.Agents[ag] = eval(agType)(Name=ag, Config=agConfig, StartDate=StartDate, DataLength=self.WS["DataLength"])
                    except Exception as e:
                        logger.error(traceback.format_exc())
                        raise Error("Fail to load {} as agent type {}.".format(ag, agType))
        # --------------------------------
        
        
        # ----- Time step simulation (Coupling hydrological model and ABM) -----
        # Obtain datetime index
        pdDatedateIndex = date_range(start = StartDate, periods = self.WS["DataLength"], freq = "D")
        self.pdDatedateIndex = pdDatedateIndex  # So users can use it directly.    
        SimSeq = self.SysPD["SimSeq"]
        AgSimSeq = self.SysPD["AgSimSeq"]
        InStreamAgents = self.SysPD["InStreamAgents"]   
        
        # Add instream agent to Q_routed. It will be populated after instream agent make their decisions.
        # InStreamAgents = ResDamAgentTypes & DamDivAgentTypes
        for isag in InStreamAgents:
            self.Q_routed[isag] = np.zeros(self.WS["DataLength"])
        
        # Run time step routing and agent simulation to update Q_LSM.
        for t in tqdm(range(self.WS["DataLength"]), desc = self.__name__, disable=disable):
            CurrentDate = pdDatedateIndex[t]
            if self.ABM is None: # Only LSM and Routing.
                for node in SimSeq:
                    if node in RoutingOutlets:
                        #----- Run Lohmann routing model for one routing outlet (node) for 1 timestep (day).
                        if self.RR["Model"] == "Lohmann":
                            Qt = runTimeStep_Lohmann(node, self.RR, self.UH_Lohmann, self.Q_routed, self.Q_LSM, t)
                        #----- Store Qt to final output.
                        self.Q_routed[node][t] = Qt 
                        
            else: # Coupled model
                for node in SimSeq:
                    RiverDivAgents_Plus = AgSimSeq["AgSimPlus"][node].get("RiverDivAgents")
                    RiverDivAgents_Minus = AgSimSeq["AgSimMinus"][node].get("RiverDivAgents")
                    InsituDivAgents_Minus = AgSimSeq["AgSimMinus"][node].get("InsituDivAgents")
                    DamDivAgents_Plus = AgSimSeq["AgSimPlus"][node].get("DamDivAgents")
                    ResDamAgents_Plus = AgSimSeq["AgSimPlus"][node].get("ResDamAgents")

                    # Note for the first three if, we should only enter one of them at each node.
                    if InsituDivAgents_Minus is not None or RiverDivAgents_Plus is not None:
                        r"""
                        For InsituDivAgents, they divert water directly from the runoff in each sub-basin or grid.
                        Note that InsituDivAgents has no return flow option.
                        After updating Q generated by LSM and plus the return flow, we run the routing to calculate the routing streamflow at the routing outlet stored in Q_routed.
                        Note that we need to use the "self.Q_LSM - Div + return flow" for the routing outlet to run the routing model.
                        That means return flow will join the in grid-routing!!
                        Therefore, we feed in both self.Q_routed and self.Q_LSM
                        """
                        if InsituDivAgents_Minus is not None:
                            for ag in InsituDivAgents_Minus:
                                self.Q_routed = self.Agents[ag].act(self.Q_routed, AgentDict = self.Agents, node=node, CurrentDate=CurrentDate, t=t)
                                # self.Q_LSM - Div
                                self.Q_LSM = self.Agents[ag].act(self.Q_LSM, AgentDict = self.Agents, node=node, CurrentDate=CurrentDate, t=t)
                                
                        if RiverDivAgents_Plus is not None:    
                            for ag in RiverDivAgents_Plus:
                                self.Q_routed = self.Agents[ag].act(self.Q_routed, AgentDict = self.Agents, node=node, CurrentDate=CurrentDate, t=t)
                                # self.Q_LSM + return flow   => return flow will join the in-grid routing. 
                                self.Q_LSM = self.Agents[ag].act(self.Q_LSM, AgentDict = self.Agents, node=node, CurrentDate=CurrentDate, t=t)
                    
                    elif ResDamAgents_Plus is not None:
                        r"""
                        For ResDamAgents, we simply add the release water to self.Q_routed[isag].
                        No minus action is needed.
                        """
                        for ag in ResDamAgents_Plus:
                            self.Q_routed = self.Agents[ag].act(self.Q_routed, AgentDict = self.Agents, node=node, CurrentDate=CurrentDate, t=t)
                    
                    elif DamDivAgents_Plus is not None:
                        r"""
                        For DamDivAgents_Plus, we simply add the release water to self.Q_routed[isag].
                        No minus action is needed.
                        Note that even the DM is diversion, the action should be converted to release (code in agent class). 
                        """
                        for ag in DamDivAgents_Plus:
                            self.Q_routed = self.Agents[ag].act(self.Q_routed, AgentDict = self.Agents, node=node, CurrentDate=CurrentDate, t=t)
                    
                    if node in RoutingOutlets:
                        #----- Run Lohmann routing model for one routing outlet (node) for 1 timestep (day).
                        if self.RR["Model"] == "Lohmann":
                            Qt = runTimeStep_Lohmann(node, self.RR, self.UH_Lohmann, self.Q_routed, self.Q_LSM, t)
                        #----- Store Qt to final output.
                        self.Q_routed[node][t] = Qt 
                        
                    
                    if RiverDivAgents_Minus is not None:
                        r"""
                        For RiverDivAgents_Minus, we divert water from the routed river flow.
                        No minus action is needed.
                        Note that even the DM is diversion, the action should be converted to release (code in agent class). 
                        """
                        for ag in RiverDivAgents_Minus:
                            self.Q_routed = self.Agents[ag].act(self.Q_routed, AgentDict = self.Agents, node=node, CurrentDate=CurrentDate, t=t)
                    

        # ----------------------------------------------------------------------
        
        logger.info("[{}] Complete HydroCNHS simulation! [{}]\n".format(self.__name__, getElapsedTime()))
        # [cms] Streamflow for routing outlets (Gauged outlets and inflow outlets of in-stream agents).
        # For other variables users need to extract them manually from this class.
        return self.Q_routed   
    
    def getModelObject(self):
        return self.__dict__
    
    
    
    
    
#%%
                # # These are not duplicated and redundant code!
                # # First, for instream agent, we don't run routing model, which we simply assign 
                # # its upstream routing outlet (routing time lag is already considerred at here) as its inflow.
                # # Second, we run Plus actions, then Minus actions.
                # if node in InStreamAgents:      # The node is an in-stream agent.
                #     #----- Update in-stream agent's actions to streamflow (self.Q_LSM) for later routing usage.
                #     for ag in AgSimSeq["AgSimPlus"][node]:
                #         self.Q_LSM = self.Agents[ag].act(self.Q_LSM, AgentDict = self.Agents, node=node, CurrentDate=CurrentDate, t=t)
                #     for ag in AgSimSeq["AgSimMinus"][node]:
                #         self.Q_LSM = self.Agents[ag].act(self.Q_LSM, AgentDict = self.Agents, node=node, CurrentDate=CurrentDate, t=t)
                # else:                           # The node is a routing outlet.
                #     if self.RR["Model"] == "Lohmann":
                #         #----- Update none in-stream agent's actions (imply in AgSimSeq calculation) to streamflow (self.Q_LSM) for later routing usage.
                #         if self.ABM is not None: 
                #             for ag in AgSimSeq["AgSimPlus"][node]:
                #                 self.Q_LSM = self.Agents[ag].act(self.Q_LSM, AgentDict = self.Agents, node=node, CurrentDate=CurrentDate, t=t)
                #             for ag in AgSimSeq["AgSimMinus"][node]:
                #                 self.Q_LSM = self.Agents[ag].act(self.Q_LSM, AgentDict = self.Agents, node=node, CurrentDate=CurrentDate, t=t)
                        
                #         #----- Run Lohmann routing model for one routing outlet (node) for 1 timestep (day). 
                #         Qt = runTimeStep_Lohmann(node, self.RR, self.UH_Lohmann, self.Q_LSM, t)
                        
                #         #----- Store Qt to final output.
                #         self.Q_routed[node][t] = Qt 