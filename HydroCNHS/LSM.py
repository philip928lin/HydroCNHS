#===================================================================================
# Land Surface model using GWLF model.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# GWLF  is based on the code from Ethan Yang @ Lehigh University (yey217@lehigh.edu)
# HYMOD is based on the code from KMarkert @ https://github.com/KMarkert/hymod
# 2021/02/05
#===================================================================================

import numpy as np
from pandas import date_range, to_datetime, to_numeric, DataFrame
import logging

r"""
# HYMOD model
Weather:
    T:                                        # [degC] Daily mean temperature.
    P:                                        # [cm] Daily precipitation.
    PE:                                       # [cm] Daily potential evapotranspiration.
Inputs:
    Area:                                     # [ha] Sub-basin area.
    SnowS:  5                                 # [cm] Snow storage.
    Latitude:                                 # [Degree] PET by Hamon.
    #s:                                        # [cm] Initial soil moisture.
    #Slow: 0                                   # [cm] Initial slow tank soil moisture.
    #Fast: [0,0,0]                             # [cm] Initial fast tanks soil moisture.
HYMODPars:                                    ## For HYMOD, we have 6 parameters.
    Cmax:                                     # [cm] Maximum storage capacity. [1, 10]
    B:                                     # Degree of spatial variability of the soil moisture capacity. [0, 2]
    Alpha:                                    # Factor distributing the flow between slow and quick release reservoirs. [0.2, 0.99]
    Kq:                                       # Residence time of the slow release reservoir. [0.5, 1.2]
    Ks:                                       # Residence time of the quick release reservoirs. [0.01, 0.5]
    Df:                                       # [cm] Snow storage.
"""
def runHYMOD(HYMODPars, Inputs, Tt, Pt, PEt, DataLength):
    """HYMOD for rainfall runoff simulation with additional snow module.
        Paper: https://piahs.copernicus.org/articles/368/180/2015/piahs-368-180-2015.pdf
        Code: https://github.com/bartnijssen/pythonlib/blob/master/hymod.py
    Args:
        HYMODPars (dict): [description]
        Inputs (dict): [description]
        Pt (Array): [cm] Daily precipitation.
        Tt (Array): [degC] Daily mean temperature.
        PEt (Array): [cm] Daily potential evaportranspiration.
        DataLength (int): Total data length.
        
    Returns:
        [Array]: [cms] CMS (Qt)
    """
    CMS = np.zeros(DataLength) # Create a 1D array to store results
    Cmax = HYMODPars["Cmax"]*10         # [cm to mm] Upper limit of ET resistance parameter [0.5, 150]*10
    B = HYMODPars["B"]            # Distribution function shape parameter [0.01,4]
    Alpha = HYMODPars["Alpha"]          # Quick-slow split parameter [0.01,0.99]
    Kq = HYMODPars["Kq"]                # [1/day] Quick flow routing tanks rate parameter [0.01,0.99]
    Ks = HYMODPars["Ks"]                # [1/day] Slow flow routing tanks rate parameter [0.001,0.01]

    Df = HYMODPars["Df"]                # Snow melt coef.
    SnowSt = Inputs["SnowS"]*10         # [cm to mm] Initial snow storage.

    Pt = np.array(Pt)*10                # [cm to mm] Daily precipitation.
    Tt = np.array(Tt)                   # [degC] Daily mean temperature.
    PEt = np.array(PEt)*10              # [cm to mm] Daily potential evapotranspiration.

    Sstar = 0 #Inputs["InitS"]*10          # [cm to mm] Initial soil moisture
    Smax = Cmax / (1. + B)

    Ss = 0                              # Initial slow tank soil moisture.
    S1 = 0                
    S2 = 0
    S3 = 0

    for t in range(DataLength): 
        # Snow module
        # Determine rainfall, snowfall and snow accumulation
        if Tt[t] > 0:           # If temperature is above 0 degC, 
            Rt = Pt[t]          # precipitation is rainfall (cm) and no snow accumulation
        else:
            Rt = 0              # Else, precipitation is snowfall (cm) so rainfall = 0
            SnowSt = SnowSt + Pt[t] # Snowfall will accumulated and become snow storage(cm)	
        # Determine snowmelt (Degree-day method)
        if Tt[t] > 0:           # Temperature above 0 degC
            Mt = min(SnowSt, Df * Tt[t])   # Snowmelt (cm) capped by snow storage
            SnowSt = SnowSt - Mt              # Update snow storage
        else:	
            Mt = 0
        P = Rt + Mt
        
        C = Cmax*(  1 - (1-((B+1)*Sstar)/(Cmax))**(1/(B+1)) )
        ER1 = max(P+C-Cmax,0)
        Cstar = min(P+C,Cmax)
        S = (Cmax/(B+1)) * (1-(1-(Cstar/Cmax))**(B+1))
        e = min(PEt[t]*Cstar/Cmax,S)
        ER2 = max((Cstar-C)-(S-Sstar),0)
        Sstar = S-e
        Ss = (1-Ks)*Ss+(1-Ks)*(1-Alpha)*ER2
        Qs = (Ks/(1-Ks))*Ss
        S1 = (1-Kq)*S1+(1-Kq)*(ER1+Alpha*ER2)
        Qq1 = (Kq/(1-Kq))*S1
        S2 = (1-Kq)*S2+(1-Kq)*Qq1
        Qq2 = (Kq/(1-Kq))*S2
        S3 = (1-Kq)*S3+(1-Kq)*Qq2
        Qq3 = (Kq/(1-Kq))*S3
        Q = Qs+Qq3
        CMS[t] = (Q * 0.001 * Inputs["Area"] * 10000) / 86400
        if S<0 | Ss<0 | S1<0 | S2<0 | S3<0 | C<0 | Cstar<0 | e<0 | ER1<0 | ER2<0 | Qs<0 | Qq1<0 | Qq2<0 | Qq3<0:
            print('infeasible')
            break
        return CMS


r"""
def runHYMOD(HYMODPars, Inputs, Tt, Pt, PEt, DataLength):
    HYMOD for rainfall runoff simulation with additional snow module.
        Paper: https://piahs.copernicus.org/articles/368/180/2015/piahs-368-180-2015.pdf
        Code: https://github.com/bartnijssen/pythonlib/blob/master/hymod.py
    Args:
        HYMODPars (dict): [description]
        Inputs (dict): [description]
        Pt (Array): [cm] Daily precipitation.
        Tt (Array): [degC] Daily mean temperature.
        PEt (Array): [cm] Daily potential evaportranspiration.
        DataLength (int): Total data length.
        
    Returns:
        [Array]: [cms] CMS (Qt)
    
    Cmax = HYMODPars["Cmax"]*10         # [cm to mm] Upper limit of ET resistance parameter
    Bexp = HYMODPars["Bexp"]            # Distribution function shape parameter
    Alpha = HYMODPars["Alpha"]          # Quick-slow split parameter
    Kq = HYMODPars["Kq"]                # Quick flow routing tanks rate parameter
    Ks = HYMODPars["Ks"]                # Slow flow routing tanks rate parameter
    Df = HYMODPars["Df"]                # Snow melt coef.
    SnowSt = Inputs["SnowS"]*10         # [cm to mm] Initial snow storage.
    
    Pt = np.array(Pt)*10                # [cm to mm] Daily precipitation.
    Tt = np.array(Tt)                   # [degC] Daily mean temperature.
    PEt = np.array(PEt)*10              # [cm to mm] Daily potential evapotranspiration.
    
    s = Inputs["s"]*10                  # cm to mm
    Smax = Cmax / (1. + Bexp)
    error = 0
    # Initialize slow tank state
    # value of 0 init flow works ok if calibration data starts with low discharge
    x_slow = Inputs["Slow"]*10              # cm to mm
    
    # Initialize state(s) of quick tank(s)
    x_quick = np.array(Inputs["Fast"])*10   # cm to mm
    CMS = np.zeros(DataLength) # Create a 1D array to store results
    
    #----- START PROGRAMMING LOOP WITH DETERMINING RAINFALL - RUNOFF AMOUNTS
    for i in range(DataLength): 
        
        # Snow module
        # Determine rainfall, snowfall and snow accumulation
        if Tt[i] > 0:           # If temperature is above 0 degC, 
            Rt = Pt[i]          # precipitation is rainfall (cm) and no snow accumulation
        else:
            Rt = 0              # Else, precipitation is snowfall (cm) so rainfall = 0
            SnowSt = SnowSt + Pt[i] # Snowfall will accumulated and become snow storage(cm)	
        # Determine snowmelt (Degree-day method)
        if Tt[i] > 0:           # Temperature above 0 degC
            Mt = min(SnowSt, Df * Tt[i])   # Snowmelt (cm) capped by snow storage
            SnowSt = SnowSt - Mt              # Update snow storage
        else:	
            Mt = 0
        
        
        # Compute excess precipitation and evaporation
        ##ER1, ER2, x_loss = calExcess(x_loss, Cmax, Bexp, Rt+Mt, PEt[i])
        
        if s > Smax:
            error += s - 0.999 * Smax
            s = 0.999 * Smax
        cprev = Cmax * (1 - np.power((1-((Bexp+1)*s/Cmax)), (1/(Bexp+1))))
        P = Rt + Mt
        ER1 = np.maximum(P + cprev - Cmax, 0.0) # effective rainfal part 1
        P -= ER1
        dummy = np.minimum(((cprev + P)/Cmax), 1)
        s1 = (Cmax/(Bexp+1)) * (1 - np.power((1-dummy), (Bexp+1))) # new state
        ER2 = np.maximum(P-(s1-s), 0) # effective rainfall part 2
        evap = np.minimum(s1, s1/Smax * PEt[i]) # actual ET is linearly related to the soil moisture state
        s = s1-evap # update state

        # Calculate total effective rainfall
        UQ = ER1 + Alpha * ER2 # quickflow contribution
        US = (1 - Alpha) * ER2 # slowflow contribution
        for i in range(3):
            x_quick[i] = (1-Kq) * x_quick[i] + (1-Kq) * UQ # forecast step
            UQ = (Kq/(1-Kq)) * x_quick[i]
        x_slow = (1-Ks) * x_slow + (1-Ks) * US
        US = (Ks/(1-Ks)) * x_slow
        Q = UQ + US
        
        # Compute total flow and convert mm to cms.
        CMS[i] = (Q * 0.001 * Inputs["Area"] * 10000) / 86400
    return CMS
"""

# The ABCD model is mainly follow (Guillermo et al., 2010); however, with 
# different snow module. 
# https://doi.org/10.1029/2009WR008294
def runABCD(ABCDPars, Inputs, Tt, Pt, PEt, DataLength):
    """ABCD for rainfall runoff simulation.

    Args:
        ABCDPars (dict): Contain 5 parameters.  
        Inputs (dict): Contain 3 inputs
        Tt (Array): [degC] Daily mean temperature.
        Pt (Array): [cm] Daily precipitation.
        PEt (Array): [cm] Daily potential evaportranspiration.
        DataLength (int): Total data length.
        
    Returns:
        [Array]: [cms] Discharge
    """
    
    r"""
    Inputs:
        Area:                   # [ha] Sub-basin area.
        XL:     10              # [cm] Initial saturated soil water content. [0, 400]
        SnowS:  5               # [cm] Snow storage.
    """
    # Data
    Tt = np.array(Tt)                   # [degC] Daily mean temperature.
    Pt = np.array(Pt)                   # [cm] Daily precipitation.
    PEt = np.array(PEt)                 # [cm] Daily potential evapotranspiration.
    
    # Variable
    SnowSt = Inputs["SnowS"]            # [cm] Initial snow storage.
    QU = 0                              # [cm] Runoff.
    QL = 0                              # [cm] Baseflow.
    XL = Inputs["XL"]                   # [cm] Initial saturated soil water content. [0, 400]
    XU = 0                              # [cm] Soil water storage (Antecedent Moisture).
    
    # Pars
    a = ABCDPars["a"]
    b = ABCDPars["b"]                   # [cm] [0, 400]
    c = ABCDPars["c"]                   # [0, 1]
    d = ABCDPars["d"]                   # [0, 1]
    Df = ABCDPars["Df"]                 # [0, 1]
    
    #----- Loop through all days (Python for loop ending needs +1) ------------
    CMS = np.zeros(DataLength) # Create a 1D array to store results
    for i in range(DataLength): 	
        
        # Determine rainfall, snowfall and snow accumulation-------------------
        if Tt[i] > 0:           # If temperature is above 0 degC, 
            Rt = Pt[i]          # precipitation is rainfall (cm) and no snow accumulation
        else:
            Rt = 0              # Else, precipitation is snowfall (cm) so rainfall = 0
            SnowSt = SnowSt + Pt[i] # Snowfall will accumulated and become snow storage(cm)
        #----------------------------------------------------------------------
        # Determine snowmelt (Degree-day method)-------------------------------
        if Tt[i] > 0:           # Temperature above 0 degC
            Mt = min(SnowSt, Df * Tt[i])   # Snowmelt (cm) capped by snow storage
            SnowSt = SnowSt - Mt              # Update snow storage
        else:	
            Mt = 0
        
        #----------------------------------------------------------------------    
        # Determin available water (P)
        P = Rt + Mt + XU
        
        #----------------------------------------------------------------------
        # ET opportunity (Guillermo et al., 2010)
        Pb = P+b
        a2 = 2*a
        In = (Pb/a2)**2 - Pb/a
        if In < 0:
            logger = logging.getLogger("HydroCNHS.ABCD")
            logger.error("Invalid parameter set. Highly likely that b is too small.")
            return None
        EO = Pb/a2 - In**0.5
        
        # EO = EO.real
        #----------------------------------------------------------------------
        # Actual evapotranspiration (E)
        E = EO * ( 1-np.exp(-PEt[i]/b) )
        E = min( PEt[i], max(0 , E) )
        XU = EO - E
        AW = P - EO
        XL = (XL + c*AW) / (1+d)
        QL = d * XL
        QU = (1-c) * AW
        Q = QL + QU
        #----------------------------------------------------------------------
        # Change unit to cms (m^3/sec)-----------------------------------------
        # Area [ha]
        CMS[i] = (Q * 0.01 * Inputs["Area"] * 10000) / 86400
    return CMS
        
        
                
def runGWLF(GWLFPars, Inputs, Tt, Pt, PEt, StartDate, DataLength):
    """GWLF for rainfall runoff simulation.

    Args:
        GWLFPars (dict): Contain 9 parameters.  
        Inputs (dict): Contain 5 inputs
        Tt (Array): [degC] Daily mean temperature.
        Pt (Array): [cm] Daily precipitation.
        PEt (Array): [cm] Daily potential evaportranspiration.
        StartDate (str): yyyy/mm/dd. 
        DataLength (int): Total data length.
        
    Returns:
        [Array]: [cms] Discharge
    """
    #----- Setup  -------------------------------------------------------------
    
    r"""
    Inputs:
        Area:                   # [ha] Sub-basin area.
        S0:     10              # [cm] Shallow saturated soil water content.
        U0:     10              # [cm] Unsaturated soil water content
        SnowS:  5               # [cm] Snow storage.
    """
    
    # Data
    Tt = np.array(Tt)           # [degC] Daily mean temperature.
    Pt = np.array(Pt)           # [cm] Daily precipitation.
    PEt = np.array(PEt)         # [cm] Daily potential evapotranspiration.
    
    # Pars
    CN2 = GWLFPars["CN2"]       # Curve number
    IS = GWLFPars["IS"]         # Interception coefficient 0.05 ~ 0.2
    Res = GWLFPars["Res"]       # Recession coefficient
    Sep = GWLFPars["Sep"]       # Deep seepage coefficient
    Alpha = GWLFPars["Alpha"]   # Baseflow coefficient (Eq 4 in Luo et al. (2012))
    Beta = GWLFPars["Beta"]     # Deep seepage coefficient (Eq 2 in Luo et al. (2012))
    Ur = GWLFPars["Ur"]         # [cm] Avaliable/Soil water capacity (Root zone)
    Df = GWLFPars["Df"]         # [cm/degC] Degree-day coefficient for snowmelt.
    Kc = GWLFPars["Kc"]         # Land cover coefficient.
    
    # Variables
    Gt =  Res*Inputs["S0"]      # [cm] Initialize saturated zone discharge to the stream.
    BFt = Gt					# [cm] Initialize baseflow to stream.
    St = Inputs["S0"]           # [cm] Initialize shallow saturated soil water content.
    Ut = Inputs["U0"]           # [cm] Initialize unsaturated soil water content.
    SnowSt = Inputs["SnowS"]    # [cm] Initial snow storage.
    AnteMois = [0, 0, 0, 0, 0]  # [cm] Define the initial Antecedent Moisture (5 days) as 0.

    


    # Calculate monthly mean temperature for distinguishing growing season purpose.
    StartDate = to_datetime(StartDate, format="%Y/%m/%d")                               # to Datetime
    pdDatedateIndex = date_range(start = StartDate, periods = DataLength, freq = "D")   # gen pd dateIndex
    MonthlyTavg = DataFrame(index = pdDatedateIndex)
    MonthlyTavg["T"] = Tt
    MonthlyTavg = MonthlyTavg.resample("MS").mean()     # e.g. 1950-1-"1" (Month Start, MS)
    # Broadcast back to daily sequence.
    # Note: Data has to longer than a month or it will show error.
    MonthlyTavg.index = [pdDatedateIndex[0]] + list(MonthlyTavg.index[1:-1]) + [pdDatedateIndex[-1]]
    MonthlyTavg = MonthlyTavg.resample('D').ffill()["T"].to_numpy()
    #--------------------------------------------------------------------------

    #----- Loop through all days (Python for loop ending needs +1) ------------
    CMS = np.zeros(DataLength) # Create a 1D array to store results
    for i in range(DataLength): 	
        # Determine the am1 and am2 values based on growing season for CN calculation.
        # Growing season (mean monthly temperature > 10 deg C).
        if MonthlyTavg[i] > 10:     # MonthlyTavg
            am1 = 3.6
            am2 = 5.3
        else:
            am1 = 1.3
            am2 = 2.8
            
        # Determine rainfall, snowfall and snow accumulation-------------------
        if Tt[i] > 0:           # If temperature is above 0 degC, 
            Rt = Pt[i]          # precipitation is rainfall (cm) and no snow accumulation
        else:
            Rt = 0              # Else, precipitation is snowfall (cm) so rainfall = 0
            SnowSt = SnowSt + Pt[i] # Snowfall will accumulated and become snow storage(cm)
        #----------------------------------------------------------------------	
        # Determine snowmelt (Degree-day method)-------------------------------
        if Tt[i] > 0:           # Temperature above 0 degC
            Mt = min(SnowSt, Df * Tt[i])   # Snowmelt (cm) capped by snow storage
            SnowSt = SnowSt - Mt              # Update snow storage
        else:	
            Mt = 0
        #----------------------------------------------------------------------
        # Calculate the Antecedent Moisture (at)-------------------------------
        AnteMois = [Rt + Mt] + AnteMois[0:4]    # Add new data and drop last data.
        at = np.sum(AnteMois)                   # Five days antecedent moisture
        #----------------------------------------------------------------------
        # CN calculation (Following GWLF2 setting)-----------------------------
        CN1 = (4.2 * CN2) / (10 - 0.058 * CN2)
        CN3 = (23 * CN2) / (10 + 0.13 * CN2)
        CN = None
        if at < am1:
            CN = CN1 + ((CN2 - CN1) / am1) * at
            
        if (am1 < at and at < am2):
            CN = CN2 + ((CN3 - CN2) / (am2 - am1)) * (at - am1)       
            
        if am2 < at: 
            CN = CN3	
        #----------------------------------------------------------------------
        # Calculate runoff (Qt)------------------------------------------------
        DSkt = (2540 / CN) - 25.4               # Detention parameter (cm)
        if (Rt + Mt) > IS * DSkt:
            Qt = (((Rt + Mt) - (IS * DSkt))**2) / ((Rt + Mt) + ((1-IS) * DSkt))
        else:
            Qt = 0	
        #----------------------------------------------------------------------
        # Calculate Evapotranspiration (Et)------------------------------------
        # Consider water stress (Ks) and land cover (Kc).
        if (Ut >= Ur*0.5):
            Ks = 1
        else:
            Ks = Ut/(Ur*0.5)
        Et = min((Ut+ Rt + Mt - Qt), Ks*Kc*PEt[i])
        #----------------------------------------------------------------------
        # Calculate Percolation (Pct) from unsaturated zone (Ut) to shallow saturated zone (St)------- 
        PCt = max((Ut+ Rt + Mt - Qt - Et - Ur), 0)
        #----------------------------------------------------------------------
        # Update unsaturated zone soil moistures (Ut)--------------------------
        Ut = Ut + Rt + Mt - Qt - Et - PCt       # Rt+Mt-Qt is infiltration
        #----------------------------------------------------------------------
        # Calculate groundwater discharge (Gt) and deep seepage (Dt)-----------
        Gt = Res * St
        Dt = Sep * St
        #----------------------------------------------------------------------
        # Update shallow saturated zone soil moistures (St)--------------------
        St = St + PCt - Gt - Dt
        #----------------------------------------------------------------------
        # Groundwater: Deep seepage loss (Dset)--------------------------------
        Dset = Beta * Dt    #Eq 2 in Luo et al. (2012)
        #----------------------------------------------------------------------
        # Groundwater: Recharge (Ret)------------------------------------------
        Ret = Dt - Dset     #Eq 3 in Luo et al. (2012)
        #----------------------------------------------------------------------
        # Groundwater:Baseflow (BFt)-------------------------------------------
        BFt = BFt * np.exp(-Alpha) + Ret * (1-np.exp(-Alpha)) #Eq 4 in Luo et al. (2012)
        #----------------------------------------------------------------------
        # Calculate streamflow (SF) and Monthly Qt, Pt, Et, Gt and SF----------
        SF = Qt + Gt + BFt #Streamflow = surface quick flow + subsurface flow + baseflow
        #----------------------------------------------------------------------			
        # Change unit to cms (m^3/sec)-----------------------------------------
        # Area [ha]
        CMS[i] = (SF * 0.01 * Inputs["Area"] * 10000) / 86400
        #----------------------------------------------------------------------
    # return the result array	
    return CMS

# More ET method code can be found at https://github.com/phydrus/PyEt 
def calPEt_Hamon(Tt, Lat, StartDate, dz = None):
    """Calculate potential evapotranspiration (PEt) with Hamon (1961) equation.

    Args:
        Tt (Array): [degC] Daily mean temperature.
        Lat (float): [deg] Latitude.
        StartDate (str): yyyy/mm/dd.
        dz (float): [m] Altitude temperature adjustment. Defaults to None.

    Returns:
        [Array]: [cm/day] PEt
    """
    Tt = np.array(Tt)
    # Altitude temperature adjustment
    if dz is not None:
        tlaps = 0.6 # Assume temperature decrease 0.6 degC for every 100 m elevation.
        Tt = Tt - tlaps*dz/100
    # Calculate Julian days
    DataLength = len(Tt)
    StartDate = to_datetime(StartDate, format="%Y/%m/%d")                         # to Datetime
    pdDatedateIndex = date_range(start = StartDate, periods = DataLength, freq = "D")  # gen pd dateIndex
    JDay = to_numeric(pdDatedateIndex.strftime('%j'))                               # convert to Julian days
    # Calculate solar declination [rad] from day of year (JDay) based on equations 24 in ALLen et al (1998).
    sol_dec = 0.4093 * np.sin(2. * 3.141592654 / 365. * JDay - 1.39)   
    Lat_rad = Lat*np.pi/180
    # Calculate sunset hour angle from latitude and solar declination [rad] based on equations 25 in ALLen et al (1998).
    omega = np.arccos(-np.tan(sol_dec) * np.tan(Lat_rad))
    # Calculate maximum possible daylight length [hr]
    dl = 24 / np.pi * omega  
    # From Prudhomme(hess, 2013) https://hess.copernicus.org/articles/17/1365/2013/hess-17-1365-2013-supplement.pdf
    PEt = (dl / 12) ** 2 * np.exp(Tt / 16)  # Slightly different from what we used to.
    PEt = np.array(PEt/10)                  # Convert from mm to cm
    PEt[np.where(Tt <= 0)] = 0              # Force PEt = 0 when temperature is below 0.
    #logger.info("[Hamon] Complete potential evapotranspiration (PEt) calculation.")
    return PEt      # [cm/day]