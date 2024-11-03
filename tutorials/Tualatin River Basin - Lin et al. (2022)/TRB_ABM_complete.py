import os
import pickle
import logging
import numpy as np
from HydroCNHS.abm import Base, read_factor

logger = logging.getLogger("ABM")
# logger can be used to log message (e.g., logger.info(msg)).

### Global variable and data
# Users can load data or define variables that are commonly available to all
# agent type classes and decision-making classes here.
this_dir, this_filename = os.path.split(__file__)  # Get this file directory.
with open(os.path.join(this_dir, "TRB_ABM_database.pickle"), "rb") as file:
    database = pickle.load(file)


"""
This is the auto-generated script template for a ABM module.
Make sure to add ABM module (this filename) to the model file
(.yaml).

Note:
Below is the list of inherited attributes for agent type
classes (AgtType) and (institutional) decision-making classes
(DMClass). These attributes will be assigned to each
initialized object in a HydroCNHS simulation. Namely, they
can be used for model design.

Agent type class (AgtType):
self.name         = agent's name.
self.config       = agent's configuration dictionary,
                  {'Attributes': ..., 'Inputs': ..., 'Pars': ...}.
self.start_date   = start date (datetime object).
self.current_date = current date (datetime object).
self.data_length  = data/simulation length.
self.t            = current timestep.
self.dc           = data collector object containing data. Routed streamflow
                    (Q_routed) is also collected at here.
self.rn_gen       = numpy random number generator.
self.agents       = a dictionary of all initialized agents,
                    {agt_name: agt object}.
self.dm           = (institutional) decision-making object if
                    DMClass or institution is assigned to the
                    agent, else None.

(Institutional) decision-making classes (DMClass):
self.name         = name of the agent or institute.
self.dc           = data collector object containing data. Routed streamflow
                    (Q_routed) is also collected at here.
self.rn_gen       = numpy random number generator.

Q_routed is the routed streamflow.

Please visit HydroCNHS manual for more examples.
https://hydrocnhs.readthedocs.io
"""


# AgtType
class Reservoir_AgtType(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def act(self, outlet):
        # Read corresponding factor of the given outlet
        factor = read_factor(self.config, outlet)

        if factor <= 0:
            logger.error("Something is not right in ResDam agent.")
        elif factor > 0:
            # Q_routed["HaggIn"][t] is the resevoir inflow at time t.
            res_t = self.dm.make_dm(
                self.dc.Q_routed["HaggIn"][self.t], self.current_date
            )
            action = res_t
        return action


# AgtType
class Diversion_AgtType(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pars = self.config["Pars"]
        self.dc.add_field(self.name, {"DivReq": [], "Diversion": [], "Shortage": []})

    def act(self, outlet):
        records = self.dc.get_field(self.name)
        # Get factor
        factor = read_factor(self.config, outlet)

        # Compute actual diversion (factor < 0) or return flow (factor >= 0)
        if factor < 0:  # Diversion
            # Make diversion request at the first day of each month
            if self.current_date.day == 1:
                div_req = self.dm.make_dm(
                    self.pars["a"], self.pars["b"], self.current_date
                )
                # Accumulate diversion request
                records["DivReq"] = records["DivReq"] + div_req

            # Apply the physical constraints for the available water at time t.
            div_req_t = records["DivReq"][self.t]
            available_water_t = self.dc.Q_routed[outlet][self.t]
            if div_req_t > available_water_t:
                shortage_t = div_req_t - available_water_t
                div_t = available_water_t
            else:
                div_t = div_req_t
                shortage_t = 0
            records["Diversion"].append(div_t)
            records["Shortage"].append(shortage_t)
            action = factor * div_t
        elif factor >= 0:  # Return flow
            div_t = records["Diversion"][self.t]
            action = factor * div_t

        return action


# AgtType
class FixedDiversion_AgtType(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pars = self.config["Pars"]
        self.dc.add_field(self.name, {"DivReq": [], "Diversion": [], "Shortage": []})

    def act(self, outlet):
        records = self.dc.get_field(self.name)
        # Get factor
        factor = read_factor(self.config, outlet)

        # Compute actual diversion (factor < 0) or return flow (factor >= 0)
        if factor < 0:  # Diversion
            # Make diversion request at the first day of each month
            if self.current_date.day == 1:
                div_req = self.dm.make_dm(self.current_date)
                # Accumulate diversion request
                records["DivReq"] = records["DivReq"] + div_req

            # Apply the physical constraints for the available water at time t.
            div_req_t = records["DivReq"][self.t]
            available_water_t = self.dc.Q_routed[outlet][self.t]
            if div_req_t > available_water_t:
                shortage_t = div_req_t - available_water_t
                div_t = available_water_t
            else:
                div_t = div_req_t
                shortage_t = 0
            records["Diversion"].append(div_t)
            records["Shortage"].append(shortage_t)
            action = factor * div_t
        elif factor >= 0:  # Return flow
            div_t = records["Diversion"][self.t]
            action = factor * div_t

        return action


# AgtType
class Pipe_AgtType(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obv_data = database["Pipe_M_median"]

    def act(self, outlet):
        # Get factor
        factor = read_factor(self.config, outlet)

        if factor <= 0:
            logger.error("Something is wrong about TRTR agent.")
        elif factor > 0:
            if self.current_date.year < 1991:
                Res_t = 0
            else:
                # Historical inputs.
                Res_t = self.obv_data[
                    self.current_date.year - 1991, self.current_date.month - 1
                ]
            action = factor * Res_t
            return action


# AgtType
class Drain_AgtType(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # We assume a linear urbanization rate. Namely, urbanized areas are
        # linearly increase from 5% to 50% of the subbasin area in 33 years.
        # We assume the urbanization will increase 50% of the orignal runoff
        # contributed by the unbanized region.
        # Therefore, the subbasin's runoff change due to the unbanization is
        # equal to unbanized_area% * 75% * original_runoff
        ini = 0.05
        end = 0.5
        interval = (end - ini) / 32
        self.urbanized_ratio = np.arange(0.05, 0.5 + interval, interval)

    def act(self, outlet):
        # Get factor
        factor = read_factor(self.config, outlet)
        Qt_change = (
            self.urbanized_ratio[self.current_date.year - 1981]
            * 0.75
            * self.dc.Q_routed[outlet][self.t]
        )
        action = factor * Qt_change
        return action


# DMClass
class ReleaseDM(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.db = database  # [m^3]
        self.flood_control = [
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
        ]
        # [cms] Use quantile 0.01 of the historical data.
        self.min_release = 0.263335  # [cms]
        self.min_release_vol = self.min_release * 86400  # [m^3]
        # Add a new field to the data collector (dc) to record storage and
        # release
        self.dc.add_field("ResAgt", {"Storage": [], "Release": []})

    def make_dm(self, inflow, current_date):
        db = self.db
        flood_control = self.flood_control
        records = self.dc.ResAgt
        min_release = self.min_release
        min_res_vol = self.min_release_vol
        day_of_year = current_date.dayofyear

        inflow_vol = inflow * 86400  # cms to m^3
        if records["Storage"] == []:  # Initial value [m^3]
            storage = 42944903.6561 + inflow_vol
        else:
            storage = records["Storage"][-1] + inflow_vol
        release = 0

        if flood_control[current_date.month - 1]:  # Flood control
            storage_max = db["Storage_q95"][day_of_year - 1]
            if storage > storage_max:
                release = (storage - storage_max) / 86400  # m^3 to cms
                storage = storage_max
            else:
                if storage - min_res_vol < 0:
                    release = 0
                else:
                    release = min_release
                    storage = storage - min_res_vol
        else:  # Target storage control
            release_target = db["Release_q50"][day_of_year - 1]
            storage_temp = storage - release_target * 86400
            if storage_temp > db["Storage_max"][day_of_year - 1]:
                release = (storage - db["Storage_max"][day_of_year - 1]) / 86400
                storage = db["Storage_max"][day_of_year - 1]
            elif storage_temp < db["Storage_q05"][day_of_year - 1]:
                release = (storage - db["Storage_q05"][day_of_year - 1]) / 86400
                storage = db["Storage_q05"][day_of_year - 1]
                if release < 0:
                    release = 0
                    storage = records["Storage"][-1] + inflow_vol
            else:
                release = release_target
                storage = storage_temp
        records["Storage"].append(storage)
        records["Release"].append(release)
        return release


# DMClass
class DivertDM(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.db = database

    def make_dm(self, a, b, current_date):
        prec_M_mean = self.db["Prec_M_mean"][
            current_date.year - 1981, (current_date.month - 1)
        ]
        div_M_mean = self.db["Div_M_median_mean"][(current_date.month - 1)]
        div_M_max = self.db["Div_M_median_max"][(current_date.month - 1)]
        div_M_min = self.db["Div_M_median_min"][(current_date.month - 1)]
        if current_date.month in [6, 7, 8, 9]:  # Major irrigation diverion months.
            div_M_req = div_M_mean + a * prec_M_mean + b
            # Bound by historical max and min
            div_M_req = min(max(div_M_req, div_M_min), div_M_max)
        else:  # Minor irrigation diversion months.
            div_M_req = div_M_mean
        # Uniformly allocate monthly diversion to daily.
        div_D_req = [div_M_req] * (current_date.days_in_month)
        return div_D_req


# DMClass
class FixedDivertDM(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.db = database

    def make_dm(self, current_date):
        div_M_mean = self.db["Div_M_median_mean"][(current_date.month - 1)]
        # Based on the historical mean value.
        div_M_req = div_M_mean
        # Uniformly allocate monthly diversion to daily.
        div_D_req = [div_M_req] * (current_date.days_in_month)
        return div_D_req


# DMClass
class TransferDM(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass
