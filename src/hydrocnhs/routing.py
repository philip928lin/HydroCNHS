# Lohmann routing module
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com)
# 2021/02/05
# Last update at 2021/12/22.

import numpy as np
from scipy.stats import gamma

# Inputs
#  FlowLength = [m] Travel distence of flow between two outlets.
#  InstreamControl = [bool] Instream control (e.g., dams).
# Parameters
#  GShape = Subbasin's UH shape parameter (gamma function).
#  GScale = Subbasin's UH scale parameter (gamma function).
#  Velo  = [m/s] Diffusion wave celerity in the linearized Saint-Venant equation.
#  Diff  = [m2/s] Diffusive coefficient in the linearized Saint-Venant equation.

# Note:
# We fix the base time setting here. For future development, it is better to
# turn entire Lohmann routing into a class with base time setting as attributes
# which allow to be changed accordingly.


def form_UH_Lohmann(inputs, routing_pars, force_ingrid_off=False):
    """Derive HRU's UH at the (watershed) outlet.

    Parameters
    ----------
    inputs : dict
        Inputs dictionary containing FlowLength [m] Travel distence of flow
        between two outlets [float] and InstreamControl [bool].
    routing_pars : dict
        Four parameters for routing: GShape, GScale, Velo, Diff [float]
    force_ingrid_off : bool, optional
        If True, then within subbasin routing will be forced to turn off, by default
        False.
    """
    flow_len = inputs["FlowLength"]
    instream_control = inputs["InstreamControl"]

    # ----- Base Time for within subbasin UH and river/channel
    # routing UH --------------------------------------------------------------
    # Within-subbasin routing
    T_IG = 12  # [day] Base time for within subbasin UH
    # River routing
    T_RR = 96  # [day] Base time for river routing UH
    dT_sec = 3600  # [sec] Time step in second for solving
    ##Saint-Venant equation. This will affect Tmax_hr.
    Tmax_hr = T_RR * 24  # [hr] Base time of river routing UH in hour
    ##because dT_sec is for an hour
    Tgr_hr = 48 * 50  # [hr] Base time for Green function values
    # --------------------------------------------------------------------------

    # ----- Within-subbasin routing UH (daily) represented by Gamma distribution
    UH_IG = np.zeros(T_IG)
    if instream_control or force_ingrid_off:
        # No time delay for within subbasin routing when the water is released
        # through instream control objects (e.g. dam).
        UH_IG[0] = 1
    elif routing_pars.get("GShape") is None and routing_pars.get("GScale") is None:
        # No time delay for within subbasin routing since Q is given by user
        # and we assume Q is observed streamflow, which no need to consider
        # time delay for within subbasin routing. This is trigger automatically
        # in HydroCNHS module.
        UH_IG[0] = 1
    else:
        shape = routing_pars["GShape"]
        scale = routing_pars["GScale"]
        if scale <= 0.0001:
            scale = 0.0001  # Since we cannot divide zero.
        for i in range(T_IG):
            # x-axis is in hr unit. We calculate in daily time step.
            UH_IG[i] = gamma.cdf(24 * (i + 1), a=shape, loc=0, scale=scale) - gamma.cdf(
                24 * i, a=shape, loc=0, scale=scale
            )
    # --------------------------------------------------------------------------

    # ----- Derive Daily River Impulse Response Function (Green's function) ----
    UH_RR = np.zeros(T_RR)
    if flow_len == 0:
        # No time delay for river routing when the outlet is gauged outlet.
        UH_RR[0] = 1
    else:
        Velo = routing_pars["Velo"]
        Diff = routing_pars["Diff"]

        # Calculate h(x, t)
        t = 0
        UH_RRm = np.zeros(Tgr_hr)  # h() in hour
        for k in range(Tgr_hr):
            # Since Velo is m/s, we use the t in sec with 1 hr as time step.
            t = t + dT_sec
            pot = ((Velo * t - flow_len) ** 2) / (4 * Diff * t)
            if pot <= 69:  # e^(-69) = E-30 a cut-off threshold
                H = flow_len / (2 * t * (np.pi * t * Diff) ** 0.5) * np.exp(-pot)
            else:
                H = 0
            UH_RRm[k] = H

        if sum(UH_RRm) == 0:
            UH_RRm[0] = 1.0
        else:
            UH_RRm = UH_RRm / sum(UH_RRm)  # Force UH_RRm to sum to 1

        # Much quicker!!  Think about S-hydrograph process.
        # And remember we should have [0] in UH_RRm[0].
        # Therefore, we use i+1 and 23 to form S-hydrolograph.
        FR = np.zeros((Tmax_hr + 23, Tmax_hr - 1))
        for i in range(Tmax_hr - 1):
            FR[:, i] = np.pad(UH_RRm, (i + 1, 23), "constant", constant_values=(0, 0))[
                : Tmax_hr + 23
            ]
        FR = np.sum(FR, axis=1) / 24
        # Lag 24 hrs
        FR = (
            FR[:Tmax_hr]
            - np.pad(FR, (24, 0), "constant", constant_values=(0, 0))[:Tmax_hr]
        )

        # Aggregate to daily UH
        for t in range(T_RR):
            UH_RR[t] = sum(FR[(24 * (t + 1) - 24) : (24 * (t + 1) - 1)])

    # ----- Combined UH_IG and UH_RR for HRU's response at the (watershed)
    # outlet ------------------------------------------------------------------
    UH_direct = np.zeros(T_IG + T_RR - 1)  # Convolute total time step [day]
    for k in range(0, T_IG):
        for u in range(0, T_RR):
            UH_direct[k + u] = UH_direct[k + u] + UH_IG[k] * UH_RR[u]
    UH_direct = UH_direct / sum(UH_direct)
    # Trim zero from back. So when we run routing, we don't need to run whole
    # array.
    UH_direct = np.trim_zeros(UH_direct, "b")
    return UH_direct


def run_step_Lohmann_sed(routing_outlet, routing, UH_Lohmann, Q, Q_runoff, t, Q_frac):
    """Calculate a single time step routing for a given routing_outlet at time t.

    Parameters
    ----------
    routing_outlet : str
        routing outlet.
    routing : dict
        Routing setting dictionary from model.yaml file.
    UH_Lohmann : dict
        A dictionary containing pre-formed UHs.
    Q : dict
        A dictionary containing newest routed flows.
    Q_runoff : dict
        A dictionary containing newest unrouted flows without.
    t : int
        Index of current time step [day].

    Returns
    -------
    float
        Routed flow of routing_outlet at time t.
    """
    Qt = None
    ro = routing_outlet
    Qresult = 0
    Subbasin = list(routing[ro].keys())
    for sb in Subbasin:
        l = len(UH_Lohmann[(sb, ro)]) - 1
        # t+1 is length, t is index.
        UH = UH_Lohmann[(sb, ro)][0 : min(t + 1, l)]
        if ro == sb:
            # Q[ro] is routed Q. We need to use unrouted Q (Q_runoff) to run the
            # routing.
            Q_reverse = np.flip(Q_runoff[sb][max(t - (l - 1), 0) : t + 1])
        else:
            Q_reverse = np.flip(Q[sb][max(t - (l - 1), 0) : t + 1])
        q_frac = np.sum(UH * Q_reverse)
        Q_frac[ro][sb][t] = q_frac
        Qresult += q_frac
    Qt = Qresult  # Store the result for time t
    return Qt


def run_step_Lohmann_convey_sed(
    routing_outlet, routing, UH_Lohmann_convey, Q_convey, t, Q_frac
):
    """Calculate a single time step conveying water routing for a given
    routing_outlet at time t.

    Parameters
    ----------
    routing_outlet : str
        routing outlet.
    routing : dict
        Routing setting dictionary from model.yaml file.
    UH_Lohmann_convey : dict
        A dictionary containing pre-formed conveying UHs (i.e., no within
        subbasin routing).
    Q_convey : dict
        A dictionary containing conveying water.
    t : int
        Index of current time step [day].

    Returns
    -------
    float
        Routed conveying flow of routing_outlet at time t.
    """
    Qt = None
    ro = routing_outlet
    Qresult = 0
    Subbasin = list(routing[ro].keys())
    for sb in Subbasin:
        uh_convey = UH_Lohmann_convey.get((sb, ro))
        if uh_convey is not None:
            l = len(uh_convey) - 1
            # t+1 is length, t is index.
            UH = uh_convey[0 : min(t + 1, l)]
            Q_reverse = np.flip(Q_convey[sb][max(t - (l - 1), 0) : t + 1])
            # Q_frac[sb][t] should already have the value from the normal routing.
            q_frac = np.sum(UH * Q_reverse)
            Q_frac[ro][sb][t] += q_frac
            Qresult += q_frac
    Qt = Qresult  # Store the result for time t
    return Qt


def run_step_Lohmann(routing_outlet, routing, UH_Lohmann, Q, Q_runoff, t, *args):
    """Calculate a single time step routing for a given routing_outlet at time t.

    Parameters
    ----------
    routing_outlet : str
        routing outlet.
    routing : dict
        Routing setting dictionary from model.yaml file.
    UH_Lohmann : dict
        A dictionary containing pre-formed UHs.
    Q : dict
        A dictionary containing newest routed flows.
    Q_runoff : dict
        A dictionary containing newest unrouted flows without.
    t : int
        Index of current time step [day].

    Returns
    -------
    float
        Routed flow of routing_outlet at time t.
    """
    Qt = None
    ro = routing_outlet
    Qresult = 0
    Subbasin = list(routing[ro].keys())
    for sb in Subbasin:
        l = len(UH_Lohmann[(sb, ro)]) - 1
        # t+1 is length, t is index.
        UH = UH_Lohmann[(sb, ro)][0 : min(t + 1, l)]
        if ro == sb:
            # Q[ro] is routed Q. We need to use unrouted Q (Q_runoff) to run the
            # routing.
            Q_reverse = np.flip(Q_runoff[sb][max(t - (l - 1), 0) : t + 1])
        else:
            Q_reverse = np.flip(Q[sb][max(t - (l - 1), 0) : t + 1])
        Qresult += np.sum(UH * Q_reverse)
    Qt = Qresult  # Store the result for time t
    return Qt


def run_step_Lohmann_convey(
    routing_outlet, routing, UH_Lohmann_convey, Q_convey, t, *args
):
    """Calculate a single time step conveying water routing for a given
    routing_outlet at time t.

    Parameters
    ----------
    routing_outlet : str
        routing outlet.
    routing : dict
        Routing setting dictionary from model.yaml file.
    UH_Lohmann_convey : dict
        A dictionary containing pre-formed conveying UHs (i.e., no within
        subbasin routing).
    Q_convey : dict
        A dictionary containing conveying water.
    t : int
        Index of current time step [day].

    Returns
    -------
    float
        Routed conveying flow of routing_outlet at time t.
    """
    Qt = None
    ro = routing_outlet
    Qresult = 0
    Subbasin = list(routing[ro].keys())
    for sb in Subbasin:
        uh_convey = UH_Lohmann_convey.get((sb, ro))
        if uh_convey is not None:
            l = len(uh_convey) - 1
            # t+1 is length, t is index.
            UH = uh_convey[0 : min(t + 1, l)]
            Q_reverse = np.flip(Q_convey[sb][max(t - (l - 1), 0) : t + 1])
            Qresult += np.sum(UH * Q_reverse)
    Qt = Qresult  # Store the result for time t
    return Qt
