import os
import pickle
import logging
import numpy as np
from HydroCNHS.abm import Base, read_factor     # import some helpers.  
logger = logging.getLogger("ABM")               # Set logger to log msg.

# =============================================================================
# ABM Global Setting
# Users can load common data that can be read by every agent.
# =============================================================================
this_dir, this_filename = os.path.split(__file__)   # Get this file directory.
with open(os.path.join(
        this_dir, "TRB_ABM_database.pickle"), "rb") as file:
    database = pickle.load(file)
# Items include in database (the pickle file):
# ['Storage_q95', 'Storage_mean', 'Storage_q05', 'Storage_max',
# 'Release_q50', 'Release_max', 
# 'Div_M_median_mean', 'Div_M_median_max', 'Div_M_median_min', 'Prec_M_mean', 
# 'Pipe_M_median']

#%% ===========================================================================
# Customized ABM Module Design Protocal
# =============================================================================

##### Base class
# This Base() class should be inherited by users when designing their agent
# type and decision-making classes. This class will load all arguments sent 
# from HydroCNHS. We show the items that will be assigned when agent
# type and decision-making instances created by HydroCNHS below. Namely, users  
# can use those items by calling "self.<item name>" when designing agent type  
# and decision-making classes.

##### Agent_type class design outline
r"""
class Agent_type(Base):                 # Inherit base class.
    #### Initialization
    def __init__(self, **kwargs):       # Copy this.
        super().__init__(**kwargs)      # Copy this.
        ### Agent_type class's available items:
        # name: agent's name.
        # config: agent's configuration dictionary the model file (.yaml).
        # start_date: datetime object.
        # data_length: length of the simulation.
        # data_collector: a container to store simulated data.
        # rn_gen: random number generator to ensure reproducibility (e.g., 
        # self.rn_gen.random()). Note that do NOT set a global random seed in 
        # this module! All type of random number should be created by "rn_gen."
        # dm: decision making object if assigned in the model file (.yaml).
        
        ### Create fields in data_collector for storing agent's outputs if 
        # needed.
        # e.g.,
        # add_field(name of the field, field type => {} or [])
        self.data_collector.add_field("field name", {}) 
        # get_field(name of the field) for further operation.
        field = self.data_collector.get_field("field name")
        
        # Other initialization that you need for this agent_type.
        # e.g., 
        self.threshold = 0.5
        
    #### Must have this act method with exact same arguments show below.
    def act(self, Q, outlet, agent_dict, current_date, t):   # Copy this
        
        # Read corresponding factor (defined in .yaml model file) of the given
        # outlet.
        factor = read_factor(self.config, outlet)
        
        # Do some calculation
        if factor > 0:      # Calculation for adding water to the outlet.
            amount = .....
            
        elif factor < 0:    # Calculation for diverting water from the outlet.
            amount = ..... 
        
        # Calculate agent's action at time t.
        action = factor * amount
        
        # This returned action will change the streamflow at the outlet by
        # new_flow = old_flow + action
        return action
"""
##### Decision-making class design outline
r"""
class Agent_DM(Base):                   # Inherit base class.
    #### Initialization
    def __init__(self, **kwargs):       # Copy this.
        super().__init__(**kwargs)      # Copy this.
        ### Decision-making class's available items:
        # start_date: datetime object.
        # data_length: length of the simulation.
        # abm: the ABM configuration dictionary from the model file (.yaml).
        # data_collector: a container to store simulated data.
        # rn_gen: random number generator to ensure reproducibility (e.g., 
        # self.rn_gen.random()). Note that do NOT set a global random seed in 
        # this module! All type of random number should be created by "rn_gen."
        
        ### Create fields in data_collector for storing agent's outputs if 
        # needed.  
        # e.g.,
        # add_field(name of the field, field type => {} or [])
        self.data_collector.add_field("field name", {}) 
        # get_field(name of the field) for further operation.
        field = self.data_collector.get_field("field name")
        
        # Other initialization that you need for this agent_type.
        # e.g., 
        self.threshold = 0.5 

    # Your customized decision-making function. You can add any arguments.
    def make_dm(self, **kwargs):
        pass
        return some_decisions
"""

##### Other design tips
# 1. Use numpy or list do the operation instead of using pandas dataframe. 
#    df.loc tend to slow down the calculation speed. 
# 2. Store only the necessary data to the data collector to control the storage
#    requirement.
# 3. We highly encourage users to follow the above design outlines. 

#%% ===========================================================================
# The Tualatin River Basin Example
# =============================================================================
class ResDM(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.database = database        # [m^3]
        self.flood_control = [True, True, True, True, True, False,
                              False, False, False, True, True, True]
        # [cms] Use quantile 0.01 of the historical data.
        self.min_release = 0.263335     
        self.min_release_vol = self.min_release * 86400     # [m^3]
        self.data_collector.add_field("ResAgt", {})
        records = self.data_collector.get_field("ResAgt")
        records["Storage"] = []
        records["Release"] = []

    def make_dm(self, inflow, current_date):
        db = self.database
        flood_control = self.flood_control
        records = self.data_collector.R1
        min_release = self.min_release
        min_res_vol = self.min_release_vol
        day_of_year = current_date.dayofyear

        inflow_vol = inflow * 86400     # cms to m^3
        if records["Storage"] == []:    # Initial value [m^3]
            storage = 42944903.6561 + inflow_vol
        else:
            storage = records["Storage"][-1] + inflow_vol
        release = 0

        if flood_control[current_date.month-1]:
            storage_max = db["Storage_q95"][day_of_year-1]
            if storage > storage_max:
                release = (storage - storage_max) / 86400 # m^3 to cms
                storage = storage_max
            else:
                if storage - min_res_vol < 0:
                    release = 0
                else:
                    release = min_release
                    storage = storage - min_res_vol
        else:   # Target storage control
            release_target = db["Release_q50"][day_of_year-1]
            storage_temp = storage - release_target * 86400
            if storage_temp > db["Storage_max"][day_of_year-1]:
                release = (storage - db["Storage_max"][day_of_year-1]) / 86400
                storage = db["Storage_max"][day_of_year-1]
            elif storage_temp < db["Storage_q05"][day_of_year-1]:
                release = (storage - db["Storage_q05"][day_of_year-1]) / 86400
                storage = db["Storage_q05"][day_of_year-1]
                if release < 0:
                    release = 0
                    storage = records["Storage"][-1] + inflow_vol
            else:
                release = release_target
                storage = storage_temp
        records["Storage"].append(storage)
        records["Release"].append(release)
        return release

class ResDam_AgType(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = self.config["Inputs"]

    def act(self, Q, outlet, agent_dict, current_date, t):
        # Read corresponding factor
        factor = read_factor(self.config, outlet)

        # Release (factor should be 1)
        if factor <= 0:
            print("Something is not right in ResDam agent.")
        elif factor > 0:
            # Q["HaggIn"][t] is the resevoir inflow at time t.
            res_t = self.dm.make_dm(Q["HaggIn"][t], current_date)
            action = res_t
            return action
        
class DivDM(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.database = database

    def make_dm(self, a, b, current_date):
        db = self.database
        prec_M_mean = db["Prec_M_mean"][current_date.year-1981,
                                        (current_date.month-1)]
        div_M_mean = db["Div_M_median_mean"][(current_date.month-1)]
        div_M_max = db["Div_M_median_max"][(current_date.month-1)]
        div_M_min = db["Div_M_median_min"][(current_date.month-1)]
        if current_date.month in [6,7,8,9]:
            div_M_req = div_M_mean + a * prec_M_mean + b
            # Bound by historical max and min
            div_M_req = min( max(div_M_req, div_M_min), div_M_max)
        else:
            div_M_req = div_M_mean
        div_D_req = [div_M_req] * (current_date.days_in_month)
        return div_D_req

class IrrDiv_AgType(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pars = self.config["Pars"]
        self.data_collector.add_field(self.name, {})
        records = self.data_collector.get_field(self.name)
        records["DivReq"] = []
        records["Diversion"] = []
        records["Shortage"] = []
        logger.info("Initialize irrigation diversion agent: {}".format(
            self.name))

    def act(self, Q, outlet, agent_dict, current_date, t):
        self.current_date = current_date
        self.t = t
        records = self.data_collector.get_field(self.name)

        # Get factor
        factor = read_factor(self.config, outlet)
        
        # For parameterized (for calibration) factor.
        if isinstance(factor, list):
            factor = self.pars[factor[0]][factor[1]]

        # Compute actual diversion or return flow
        if factor < 0:  # Diversion
            # Make diversion request at 1st of each month
            if current_date.day == 1:
                a = self.pars["a"]
                b = self.pars["b"]
                div_req = self.dm.make_dm(a, b, current_date)
                records["DivReq"] = records["DivReq"] + div_req

            div_req_t = records["DivReq"][t]
            available_water_t = Q[outlet][t]
            if div_req_t > available_water_t:
                shortage_t = div_req_t - available_water_t
                div_t = available_water_t
            else:
                div_t = div_req_t
                shortage_t = 0
            records["Diversion"].append(div_t)
            records["Shortage"].append(shortage_t)
            action = factor * div_t
        elif factor >= 0:   # factor > 0; Return flow
            div_t = records["Div"][t]
            action = factor * div_t
        
        return action

class PipeDM(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

class Pipe_AgType(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.assigned_behavior = database["TRTR"]

    def act(self, Q, outlet, agent_dict, current_date, t):
        
        # Get factor
        factor = read_factor(self.config, outlet)

        # Release (factor should be 1)
        if factor <= 0:
            print("Something is not right in TRTR agent.")
        elif factor > 0:
            # Assuming that diversion has beed done, get the actual release at
            # time t.
            y = current_date.year
            m = current_date.month
            
            if y < 1991:
                Res_t = 0
            else:
                Res_t = self.assigned_behavior[y-1991, m-1]
            action = factor * Res_t
            return action

class Urban_AgType(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # We assume a linear urbanization rate. Namely, urbanized are linearly
        # increase from 5% to 50% of the subbasin area in 33 years.
        # We assume the urbanization will increase 50% of the orignal runoff
        # contributed by the unbanized region.
        # Therefore, the subbasin's runoff change due to the unbanization is
        # equal to unbanized_area% * 75% * original_runoff
        ini = 0.05
        end = 0.5
        interval = (end-ini)/32
        self.urbanized_ratio = np.arange(0.05, 0.5 + interval, interval)
        self.rn = self.data_collector.add_field(self.name, [])
        
    def act(self, Q, outlet, agent_dict, current_date, t):
        # Get factor
        factor = read_factor(self.config, outlet)
        Qt_change = self.urbanized_ratio[current_date.year-1981] * 0.75 \
                    * Q[outlet][t]
        action = factor * Qt_change
        
        #### Test rn_gen        
        rn = self.data_collector.get_field(self.name)
        rn.append(self.rn_gen.random())
        return action