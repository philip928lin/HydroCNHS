#%%
# Routing model for Land Surface model outputs based on Lohmann routing model
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# based on the code from Ethan Yang @ Lehigh University (yey217@lehigh.edu)
# 2021/02/05

import numpy as np
from scipy.stats import gamma
#import logging
#logger = logging.getLogger("HydroCNHS.RR") # Get logger for logging msg.

# Inputs
#  FlowLen = [m] Travel distence of flow between two outlets.

# Parameters
#  RoutePars["GShape"] = Sub-basin's UH shape parameter (Gamma distribution argument) (N)
#  RoutePars["GScale"] = Sub-basin's UH scale parameter (reservior storage constant)  (K)
#  RoutePars["Velo"]   = [m/s] Wave velocity in the linearized Saint-Venant equation   
#  RoutePars["Diff"]   = [m2/s] Diffusivity in the linearized Saint-Venant equation

# Note:
# We fix the Base Time setting here. For future development, it is better to ture whole Lohmann routing into a class with Base Time setting as initial variables which allow to be changed accordingly.

def formUH_Lohmann(Inputs, RoutePars):
    """Derive HRU's UH at the (watershed) outlet.
    We seperately calculate in-grid UH_IG and river routing UH_RR and combine them into HRU's UH.

    Args:
        Inputs (dict): Two inputs for routing: FlowLength [m] Travel distence of flow between two outlets [float], InStreamControl [bool].
        RoutePars (dict): Four parameters for routing: GShape, GScale, Velo, Diff [float]
    """
    FlowLen = Inputs["FlowLen"]
    InStreamControl = Inputs["InStreamControl"]
    
    #----- Base Time for in-grid (watershed subunit) UH and river/channel routing UH ------
    # In-grid routing
    T_IG = 12					# [day] Base time for in-grid UH 
    # River routing 
    T_RR = 96					# [day] Base time for river routing UH 
    dT_sec = 3600				# [sec] Time step in second for solving Saint-Venant equation. This will affect Tmax_hr.
    Tmax_hr = T_RR * 24			# [hr] Base time of river routing UH in hour because dT_sec is for an hour
    Tgr_hr = 48 * 50			# [hr] Base time for Green function values
    #--------------------------------------------------------------------------------------
    
    #----- In-grid routing UH (daily) represented by Gamma distribution -------------------
    UH_IG = np.zeros(T_IG)
    if InStreamControl:
        UH_IG[0] = 1    # No time delay for river routing when the water is released through in-stream control objects (e.g. reservoir).
    else:
        Shape = RoutePars["GShape"]
        Scale = RoutePars["GScale"]
        for i in range(T_IG):
            # x-axis is in hr unit. We calculate in daily time step.
            UH_IG[i] = gamma.cdf(24*(i + 1), a = Shape, loc = 0, scale = Scale) \
                        - gamma.cdf(24*i, a = Shape, loc = 0, scale = Scale)
    #--------------------------------------------------------------------------------------

    #----- Derive Daily River Impulse Response Function (Green's function) ----------------
    UH_RR = np.zeros(T_RR)
    if FlowLen == 0:
        UH_RR[0] = 1    # No time delay for river routing when the outlet is gauged outlet.
    else:
        Velo = RoutePars["Velo"] 
        Diff = RoutePars["Diff"]

        # Calculate h(x, t)
        t = 0
        UH_RRm = np.zeros(Tgr_hr) # h() in hour
        for k in range(Tgr_hr):
            t = t + dT_sec          # Since Velo is m/s, we use the t in sec with 1 hr as time step.
            pot = ((Velo * t - FlowLen) ** 2) / (4 * Diff * t)
            if pot <= 69:		    # e^(-69) = E-30 a cut-off threshold								
                H = FlowLen /(2 * t * (np.pi * t * Diff)**0.5) * np.exp(-pot)  
            else:
                H = 0
            UH_RRm[k] = H

        if sum(UH_RRm) == 0:
            UH_RRm[0] = 1.0
        else:
            UH_RRm = UH_RRm / sum(UH_RRm)     # Force UH_RRm to sum to 1
        
        # FR: Fast response flow in hourly segments.        
        FR = np.zeros((Tmax_hr, 2))     												
        FR[0:23, 0] = 1 / 24    # Later sum over 24 hours, so will need to be divided by 24.

        # S-map Unit conversion, from hr to day
        for t in range(Tmax_hr):
            for L in range (0, Tmax_hr+24):
                if (t-L) > 0:
                    FR[t,1] = FR[t,1] + FR[t-L,0] * UH_RRm[L]
        # Aggregate to daily UH
        for t in range(T_RR):
            UH_RR[t] = sum(FR[(24*(t+1)-24):(24*(t+1)-1),1])

	#----- Combined UH_IG and UH_RR for HRU's response at the (watershed) outlet ----------
    UH_direct = np.zeros(T_IG + T_RR - 1)   # Convolute total time step [day]
    for k in range (0, T_IG):
        for u in range (0, T_RR):
            UH_direct[k+u] = UH_direct[k+u] + UH_IG[k] * UH_RR[u]
    UH_direct = UH_direct/sum(UH_direct)
    #--------------------------------------------------------------------------------------
    #logger.debug("[Lohmann] Complete calculating HRU's UH for flow routing simulation.")
    return UH_direct

def runTimeStep_Lohmann(RoutingOutlet, Routing, UH_Lohmann, Q, t):
    """Calculate a single time step routing for the entire basin.
    Args:
        Routing (dict): Sub-model dictionary from your model.yaml file.
        UH_Lohmann (dict): Contain all pre-formed UH for all connections between gauged outlets and its upstream outlets. e.g. {(subbasin, gaugedoutlet): UH_direct}
        Q (dict): Contain all updated Q (array) for each outlet. 
        t (int): Index of current time step (day).

    Returns:
        [dict]: Update Qt for routing.
    """   
    #----- Base Time for in-grid (watershed subunit) UH and river/channel routing UH ------
    # In-grid routing
    T_IG = 12					# [day] Base time for in-grid UH 
    # River routing 
    T_RR = 96					# [day] Base time for river routing UH 
    #--------------------------------------------------------------------------------------
    Qt = None
    ro = RoutingOutlet
    #logger.debug("Start updating {} outlet = {} for routing at time step {}.".format(g, Q[g][t], t))
    Qresult = 0
    Subbasin = list(Routing[ro].keys())
    for sb in Subbasin:
        for j in range(T_IG + T_RR - 1):
            # Sum over the flow contributed from upstream outlets.
            if (t-j+1) >= 1:
                Qresult = Qresult + UH_Lohmann[(sb, ro)][j]*Q[sb][t-j]
    Qt = Qresult         # Store the result for time t
    #logger.debug("Complete {} outlet = {} simulation for routing at time step {}.".format(g, Qt[g], t))
    #logger.debug("Complete routing at time step {}.".format(t))
    return Qt


# def runTimeStep_Lohmann(RoutingOutlets, Routing, UH_Lohmann, Q, t):
#     """Calculate a single time step routing for the entire basin.
#     Args:
#         Routing (dict): Sub-model dictionary from your model.yaml file.
#         UH_Lohmann (dict): Contain all pre-formed UH for all connections between gauged outlets and its upstream outlets. e.g. {(subbasin, gaugedoutlet): UH_direct}
#         Q (dict): Contain all updated Q (array) for each outlet. 
#         t (int): Index of current time step (day).

#     Returns:
#         [dict]: Update Qt for routing.
#     """   
#     #----- Base Time for in-grid (watershed subunit) UH and river/channel routing UH ------
#     # In-grid routing
#     T_IG = 12					# [day] Base time for in-grid UH 
#     # River routing 
#     T_RR = 96					# [day] Base time for river routing UH 
#     #--------------------------------------------------------------------------------------
#     Qt = {}
#     for ro in RoutingOutlets:
#         #logger.debug("Start updating {} outlet = {} for routing at time step {}.".format(g, Q[g][t], t))
#         Qresult = 0
#         Subbasin = list(Routing[ro].keys())
#         for sb in Subbasin:
#             for j in range(T_IG + T_RR - 1):
#                 # Sum over the flow contributed from upstream outlets.
#                 if (t-j+1) >= 1:
#                     Qresult = Qresult + UH_Lohmann[(sb, ro)][j]*Q[sb][t-j]
#         Qt[ro] = Qresult         # Store the result for time t
#         #logger.debug("Complete {} outlet = {} simulation for routing at time step {}.".format(g, Qt[g], t))
#     #logger.debug("Complete routing at time step {}.".format(t))
#     return Qt

#%% Test function
r"""
RoutePars = {}
RoutePars["GShape"] = 62.6266
RoutePars["GScale"] = 1/0.4447
RoutePars["Velo"] = 19.1643
RoutePars["Diff"] = 1985.4228
FlowLen = 11631
UH = formUH_Lohmann(FlowLen, RoutePars)
np.sum(UH)
"""
# %%
