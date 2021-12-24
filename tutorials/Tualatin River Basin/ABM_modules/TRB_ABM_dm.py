import os
import pickle
import logging
import numpy as np
logger = logging.getLogger("ABM")

# Get this file directory.
this_dir, this_filename = os.path.split(__file__)
with open(os.path.join(
        this_dir, "TRB_ABM_database.pickle"), "rb") as file:
    database = pickle.load(file)

class base():
    def __init__(self, **kwargs):
        # Agent
        # name=agG, config=ag_config, start_date=start_date,
        # data_length=data_length, data_collector=dc, rn_gen=rn_gen
        # and assign dm or None
        
        # dm
        # start_date=start_date, data_length=data_length, abm=abm,
        # data_collector=dc, rn_gen=rn_gen
        for key in kwargs:  # Load back all the previous class attributions.
            setattr(self, key, kwargs[key])

class ResDM(base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.database = database    # [m^3]
        self.flood_control = [True, True, True, True, True, False,
                              False, False, False, True, True, True]
        # Report 53,323 acre-feet but not the observed max. We used oberseved max.
        #self.capacity = 66385375.75588767 # [m^3]
        self.min_release = 0.2633466733056  # [cms] obv_D["SCOO"].quantile(0.01)
        self.min_release_vol = self.min_release * 86400 # m^3
        self.data_collector.add_field("R1", {})
        records = self.data_collector.get_field("R1")
        records["storage"] = []
        records["release"] = []

    def make_dm(self, inflow, current_date):
        db = self.database
        flood_control = self.flood_control
        records = self.data_collector.R1
        min_release = self.min_release
        min_res_vol = self.min_release_vol
        day_of_year = current_date.dayofyear

        inflow_vol = inflow * 86400 # cms to m^3
        if records["storage"] == []:  # Initial value [m^3]
            storage = 42944903.65605376 + inflow_vol
        else:
            storage = records["storage"][-1] + inflow_vol
        release = 0

        if flood_control[current_date.month-1]:
            storage_max = db["SCO_q95"][day_of_year-1]
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
            release_target = db["SCOO_q50"][day_of_year-1]
            storage_temp = storage - release_target * 86400
            if storage_temp > db["SCO_max"][day_of_year-1]:
                release = (storage - db["SCO_max"][day_of_year-1]) / 86400
                storage = db["SCO_max"][day_of_year-1]
            elif storage_temp < db["SCO_q05"][day_of_year-1]:
                release = (storage - db["SCO_q05"][day_of_year-1]) / 86400
                storage = db["SCO_q05"][day_of_year-1]
                if release < 0:
                    release = 0
                    storage = records["storage"][-1] + inflow_vol
            else:
                release = release_target
                storage = storage_temp
        records["storage"].append(storage)
        records["release"].append(release)
        return release

class ResDam_AgType(base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = self.config["Inputs"]
        self.attributes = self.config.get("Attributes")
        self.pars = self.config["Pars"]
        self.dm_class_name = self.config["Inputs"]["DMClass"]
        self.current_date = None             # Datetime object.
        self.t = None                       # Current time step index.

    def act(self, Q, outet, agent_dict, current_date, t):
        self.agent_dict = agent_dict
        self.current_date = current_date
        self.t = t

        factor = self.inputs["Links"][outet]

        # Release (factor should be 1)
        if factor < 0:
            print("Something is not right in ResDam agent.")

        elif factor > 0:
            # Q["SCOO"][t] is the resevoir inflow
            res_t = self.dm.make_dm(Q["SCOO"][t], current_date)
            action = res_t
            return action

class DivDM(base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.database = database

    def make_dm(self, a, b, current_date):
        db = self.database
        prec_M_mean = db["prec_M_mean"][current_date.year-1981,
                                        (current_date.month-1)]
        div_M_mean = db["SHPP_M_median_mean"][(current_date.month-1)]
        div_M_max = db["SHPP_M_median_max"][(current_date.month-1)]
        div_M_min = db["SHPP_M_median_min"][(current_date.month-1)]
        if current_date.month in [6,7,8,9]:
            div_M_req = div_M_mean + a*prec_M_mean + b
            # Bound by history max and min
            div_M_req = min( max(div_M_req, div_M_min), div_M_max)
        else:
            div_M_req = div_M_mean
        div_D_req = [div_M_req] * (current_date.days_in_month)
        return div_D_req


class IrrDiv_AgType(base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = self.config["Inputs"]
        self.dm_class_name = self.config["Inputs"]["DMClass"]
        self.attributes = self.config.get("Attributes")
        self.pars = self.config["Pars"]
        self.data_collector.add_field(self.name, {})
        records = self.data_collector.get_field(self.name)
        records["DivReq"] = []
        records["Div"] = []
        records["Shortage"] = []
        logger.info("Initialize irrigation diversion agent: {}".format(
            self.name))

    def act(self, Q, outet, agent_dict, current_date, t):
        self.current_date = current_date
        self.t = t
        records = self.data_collector.get_field(self.name)

        # Get factor
        factor = self.inputs["Links"][outet]
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
            available_water_t = Q[outet][t]
            if div_req_t > available_water_t:
                shortage_t = div_req_t - available_water_t
                div_t = available_water_t
            else:
                div_t = div_req_t
                shortage_t = 0
            records["Div"].append(div_t)
            records["Shortage"].append(shortage_t)
            action = factor * div_t
        else:           # Return flow
            div_t = records["Div"][t]
            action = factor * div_t
        
        return action

class PipeDM(base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

class Pipe_AgType(base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = self.config["Inputs"]
        self.attributes = self.config.get("Attributes")
        self.pars = self.config["Pars"]

        self.current_date = None             # Datetime object.
        self.t = None                       # Current time step index.
        self.Q = None                       # Input outlets' flows.
        self.assigned_behavior = database["TRTR"]

    def act(self, Q, outet, agent_dict, current_date, t):
        self.agent_dict = agent_dict
        self.current_date = current_date
        self.t = t
        factor = self.inputs["Links"][outet]

        # Release (factor should be 1)
        if factor < 0:
            print("Something is not right in TRTR agent.")

        elif factor > 0:
            # Assume that diversion has beed done in t.
            y = current_date.year
            m = current_date.month
            if y < 1991:
                Res_t = 0
            else:
                Res_t = self.assigned_behavior[y-1991, m-1]
            action = factor * Res_t
            return action

class Urban_AgType(base):
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
        self.inputs = self.config["Inputs"]
        self.rn = self.data_collector.add_field(self.name, [])
        
    def act(self, Q, outet, agent_dict, current_date, t):
        self.agent_dict = agent_dict
        self.current_date = current_date
        self.t = t
        factor = self.inputs["Links"][outet]
        Qt_change = self.urbanized_ratio[current_date.year-1981] * 0.75 \
                    * Q[outet][t]
        action = factor * Qt_change
        
        #### Test rn_gen        
        rn = self.data_collector.get_field(self.name)
        rn.append(self.rn_gen.random())
        return action