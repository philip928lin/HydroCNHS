import numpy as np
from pandas import DataFrame
def cal_usle(RE, K, CP, LS, Area):
    """Calculate the erosion using USLE.

    Parameters
    ----------
    RE : 1darray
        Rainfall erosivity.
    K : 1darray
        Soil erodibility factor.
    CP : 1darray
        Vegetation management factor = cover and management factor * support
        practice factor.
    LS : 1darray
        Topographic factor.
    Area : 1darray
        [ha] Area

    Returns
    -------
    2darray
        [Mg] Sediment supply
    """
    RE = np.array(RE).reshape((-1,1))
    K = np.array(K).reshape((-1,1))
    CP = np.array(CP).reshape((-1,1))
    LS = np.array(LS).reshape((-1,1))
    Area = np.array(Area).reshape((-1,1))
    X = 0.132 * np.dot(K*CP*LS*Area, RE.T)
    return X    # k x t

def cal_LS(SL, PS):
    """Calculate topographic factor, LS.

    Parameters
    ----------
    SL : 1darray
        Slope length.
    PS : 1darray
        Percent slope.

    Returns
    -------
    1darray
        LS
    """
    SL = np.array(SL)
    PS = np.array(PS)
    if PS >= 5:
        b = 0.5
    elif PS < 5 and PS > 3:
        b = 0.4
    elif PS <= 3 and PS >= 1:
        b = 0.3
    elif PS <1:
        b = 0.2
    theta = np.arctan(PS/100)
    LS = (0.045*SL)**b * (65.41*np.sin(theta)**2 + 4.56*np.sin(theta) + 0.065)
    return LS   # k x 1

def cal_SX(DR, X, pd_date_index):
    """Calculate total sediment supply, SX, of subbasins.

    Parameters
    ----------
    DR : 1darray
        Delivery ratio.
    X : 2darray
        Sediment supply.
    pd_date_index : pd.Datetime
        Pandas datetime index.

    Returns
    -------
    1darray
        SX
    """
    DR = np.array(DR)
    SX = DR * DataFrame(X.T, index=pd_date_index).sum(axis=1).resample("MS").sum()
    return SX.to_numpy(), SX.index # 1 x t

# calculate entire set for a routing node
def cal_TR_B(Q_frac, pd_date_index, fi, ti):
    """Calculate monthly allocation propotions based on transportation capacity.

    Parameters
    ----------
    Q_frac : dict
        A dictionary of streamflow portion of each subbasion that contributes 
        their corresponding routing outlets.
    pd_date_index : pd.Datetime
        Pandas datetime index.

    Returns
    -------
    dict
        A dictionary of monthly allocation ratios.
    """
    
    # Note that Q_frac.keys() is the outlets involve in the routing + DamAPI 
    # agents.
    
    # Transport factor, 
    TR = DataFrame(Q_frac, index=pd_date_index)
    TR = TR.iloc[fi:ti+1,:]**(5/3)
    TR = TR.resample("MS").sum()
    # The total transport capacity, sb x t. A reversed cumulative sum. 
    B = TR.loc[::-1, :].cumsum()[::-1]
    
    ratio_dict = {m: TR.iloc[:m+1,:]/B.iloc[m,:] for m in range(12)}
    
    return ratio_dict # {t x sb}

def run_TSS(RE_dict, sediment_setting, routing_setting,
            dam_agts, pd_date_index, Q_frac, dc_TSS, yi, fi, ti): 
    # instream object they need to manually assign value of Y
    outlets = list(sediment_setting.keys())
    SX = []
    for sb in outlets:
        inputs = sediment_setting[sb]["Inputs"]
        pars = sediment_setting[sb]["Pars"]
        RE = RE_dict[sb][fi:ti+1]
        #LS = cal_LS(pars["SL"], pars["PS"])         # k x 1
        LS = pars["LS"]
        CP = pars["CP"]
        if isinstance(CP, (np.ndarray, np.generic, list)) is False:
            CP = [CP]*len(LS)
        X = cal_usle(RE, pars["K"], CP, LS,
                     inputs["Area"])   # k x t
        SX_sb, pd_date_index_m = cal_SX(pars["DR"], X, pd_date_index[fi:ti+1])   
        SX.append(SX_sb)
    
    # Add DamAgts' month sediment yield (they have to provide this without knowing
    # upstream sediment information)
    dc_TSS    # monthly
    for agt in dam_agts:
        SX.append(dc_TSS[agt][yi:yi+12])
        outlets.append(agt)
    SX = DataFrame(np.array(SX).T, index=pd_date_index_m, columns=outlets)
    #SX = np.array(SX)   # t x sb

    # Transportation capacity
    ratio_dict = cal_TR_B(Q_frac, pd_date_index, fi, ti) # t x sb
    
    # Route sediment yield.
    for m in range(12):
        ratio = ratio_dict[m]
        for ro in list(routing_setting.keys()):
            o_list = list(routing_setting[ro].keys())
            sx = SX[o_list].to_numpy()
            ra = ratio[o_list].to_numpy()
            ro_y = (sx[:m+1,:] * ra[:m+1,:]).sum()
            
            # add cap here if required
            
            dc_TSS[ro][yi*12+m] = ro_y
            # Update SX for downstream outlet routing
            SX.iloc[m, outlets.index(ro)]
    return None