#==============================================================================
# Land Surface model using GWLF model.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# GWLF  is based on the code from 
# Ethan Yang @ Lehigh University (yey217@lehigh.edu)
# HYMOD is based on the code from 
# KMarkert @ https://github.com/KMarkert/hymod
# 2021/02/05
#==============================================================================

import numpy as np
from pandas import date_range, to_datetime, to_numeric, DataFrame
import logging

# The ABCD model is mainly follow (Guillermo et al., 2010); however, with 
# different snow module. 
# https://doi.org/10.1029/2009WR008294
def runABCD(ABCD_pars, inputs, temp, prec, pet, data_length):
    """ABCD for rainfall runoff simulation.

    Args:
        ABCD_pars (dict): Contain 5 parameters.  
        inputs (dict): Contain 3 inputs
        temp (Array): [degC] Daily mean temperature.
        prec (Array): [cm] Daily precipitation.
        pet (Array): [cm] Daily potential evaportranspiration.
        data_length (int): Total data length.
        
    Returns:
        [Array]: [cms] Discharge
    """
    
    r"""
    Inputs:
        Area:                   # [ha] Sub-basin area.
        XL:     10              # [cm] Initial saturated soil water content.
                                       [0, 400]
        SnowS:  5               # [cm] Snow storage.
    """
    # Data
    temp = np.array(temp)      # [degC] Daily mean temperature.
    prec = np.array(prec)      # [cm] Daily precipitation.
    pet = np.array(pet)        # [cm] Daily potential evapotranspiration.
    
    # Variable
    SnowSt = inputs["SnowS"]   # [cm] Initial snow storage.
    QU = 0                     # [cm] Runoff.
    QL = 0                     # [cm] Baseflow.
    XL = inputs["XL"]          # [cm] Initial saturated soil water content.
                               ## [0, 400]
    XU = 0                     # [cm] Soil water storage (Antecedent Moisture).
    
    # Pars
    a = ABCD_pars["a"]
    b = ABCD_pars["b"]                   # [cm] [0, 400]
    c = ABCD_pars["c"]                   # [0, 1]
    d = ABCD_pars["d"]                   # [0, 1]
    Df = ABCD_pars["Df"]                 # [0, 1]
    
    #----- Loop through all days (Python for loop ending needs +1) ------------
    CMS = np.zeros(data_length) # Create a 1D array to store results
    for i in range(data_length): 	
        
        # Determine rainfall, snowfall and snow accumulation-------------------
        if temp[i] > 0:           # If temperature is above 0 degC, 
            # precipitation is rainfall (cm) and no snow accumulation
            Rt = prec[i]          
        else:
            # Else, precipitation is snowfall (cm) so rainfall = 0
            Rt = 0
            # Snowfall will accumulated and become snow storage(cm)
            SnowSt = SnowSt + prec[i]
        #----------------------------------------------------------------------
        # Determine snowmelt (Degree-day method)-------------------------------
        if temp[i] > 0:           # Temperature above 0 degC
            # Snowmelt (cm) capped by snow storage
            Mt = min(SnowSt, Df * temp[i])   
            SnowSt = SnowSt - Mt  # Update snow storage
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
            logger.error("Invalid parameter set. Highly likely that b is "
                         +"too small.")
            return None
        EO = Pb/a2 - In**0.5
        
        # EO = EO.real
        #----------------------------------------------------------------------
        # Actual evapotranspiration (E)
        E = EO * ( 1-np.exp(-pet[i]/b) )
        E = min( pet[i], max(0 , E) )
        XU = EO - E
        AW = P - EO
        XL = (XL + c*AW) / (1+d)
        QL = d * XL
        QU = (1-c) * AW
        Q = QL + QU
        #----------------------------------------------------------------------
        # Change unit to cms (m^3/sec)-----------------------------------------
        # Area [ha]
        CMS[i] = (Q * 0.01 * inputs["Area"] * 10000) / 86400
    return CMS
                
def runGWLF(GWLF_pars, inputs, temp, prec, pet, start_date, data_length):
    """GWLF for rainfall runoff simulation.

    Args:
        GWLF_pars (dict): Contain 9 parameters.  
        inputs (dict): Contain 5 inputs
        temp (Array): [degC] Daily mean temperature.
        prec (Array): [cm] Daily precipitation.
        pet (Array): [cm] Daily potential evaportranspiration.
        start_date (str): yyyy/mm/dd. 
        data_length (int): Total data length.
        
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
    temp = np.array(temp)           # [degC] Daily mean temperature.
    prec = np.array(prec)           # [cm] Daily precipitation.
    pet = np.array(pet)         # [cm] Daily potential evapotranspiration.
    
    # Pars
    CN2 = GWLF_pars["CN2"]     # Curve number
    IS = GWLF_pars["IS"]       # Interception coefficient 0.05 ~ 0.2
    Res = GWLF_pars["Res"]     # Recession coefficient
    Sep = GWLF_pars["Sep"]     # Deep seepage coefficient
    Alpha = GWLF_pars["Alpha"] # Baseflow coefficient 
                               ##(Eq 4 in Luo et al. (2012))
    Beta = GWLF_pars["Beta"]   # Deep seepage coefficient
                               ##(Eq 2 in Luo et al. (2012))
    Ur = GWLF_pars["Ur"]       # [cm] Avaliable/Soil water capacity
                               ##(Root zone)
    Df = GWLF_pars["Df"]       # [cm/degC] Degree-day coefficient for snowmelt.
    Kc = GWLF_pars["Kc"]       # Land cover coefficient.
    
    # Variables
    Gt =  Res*inputs["S0"]      # [cm] Initialize saturated zone discharge to
                                ##the stream.
    BFt = Gt					# [cm] Initialize baseflow to stream.
    St = inputs["S0"]           # [cm] Initialize shallow saturated soil water
                                ##content.
    Ut = inputs["U0"]           # [cm] Initialize unsaturated soil water
                                ##content.
    SnowSt = inputs["SnowS"]    # [cm] Initial snow storage.
    AnteMois = [0, 0, 0, 0, 0]  # [cm] Define the initial Antecedent Moisture
                                ##(5 days) as 0.

    # Calculate monthly mean temperature for distinguishing growing season
    # purpose.
    start_date = to_datetime(start_date, format="%Y/%m/%d")                               
    pdDatedateIndex = date_range(start=start_date, periods=data_length,
                                 freq="D")
    MonthlyTavg = DataFrame(index=pdDatedateIndex)
    MonthlyTavg["T"] = temp
    MonthlyTavg = MonthlyTavg.resample("MS").mean()
    # Broadcast back to daily sequence.
    # Note: Data has to longer than a month or it will show error.
    MonthlyTavg.index = [pdDatedateIndex[0]] + list(MonthlyTavg.index[1:-1]) \
                        + [pdDatedateIndex[-1]]
    MonthlyTavg = MonthlyTavg.resample('D').ffill()["T"].to_numpy()
    #--------------------------------------------------------------------------

    #----- Loop through all days (Python for loop ending needs +1) ------------
    CMS = np.zeros(data_length) # Create a 1D array to store results
    for i in range(data_length): 	
        # Determine the am1 and am2 values based on growing season for CN
        # calculation.
        # Growing season (mean monthly temperature > 10 deg C).
        if MonthlyTavg[i] > 10:     # MonthlyTavg
            am1 = 3.6
            am2 = 5.3
        else:
            am1 = 1.3
            am2 = 2.8
            
        # Determine rainfall, snowfall and snow accumulation-------------------
        if temp[i] > 0:           # If temperature is above 0 degC, 
            Rt = prec[i]          # precipitation is rainfall (cm) and no snow
                                  ##accumulation
        else:
            Rt = 0      # Else, precipitation is snowfall (cm) so rainfall = 0
            # Snowfall will accumulated and become snow storage(cm)
            SnowSt = SnowSt + prec[i]
        #----------------------------------------------------------------------	
        # Determine snowmelt (Degree-day method)-------------------------------
        if temp[i] > 0:           # Temperature above 0 degC
            # Snowmelt (cm) capped by snow storage
            Mt = min(SnowSt, Df * temp[i])
            SnowSt = SnowSt - Mt  # Update snow storage
        else:	
            Mt = 0
        #----------------------------------------------------------------------
        # Calculate the Antecedent Moisture (at)-------------------------------
        AnteMois = [Rt + Mt] + AnteMois[0:4] # Add new data and drop last data.
        at = np.sum(AnteMois)                # Five days antecedent moisture
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
        Et = min((Ut+ Rt + Mt - Qt), Ks*Kc*pet[i])
        #----------------------------------------------------------------------
        # Calculate Percolation (Pct) from unsaturated zone (Ut) to shallow
        # saturated zone (St)--------------------------------------------------
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
        # Eq 4 in Luo et al. (2012)
        BFt = BFt * np.exp(-Alpha) + Ret * (1-np.exp(-Alpha))
        #----------------------------------------------------------------------
        # Calculate streamflow (SF) and Monthly Qt, prec, Et, Gt and SF--------
        # Streamflow = surface quick flow + subsurface flow + baseflow
        SF = Qt + Gt + BFt
        #----------------------------------------------------------------------			
        # Change unit to cms (m^3/sec)-----------------------------------------
        # Area [ha]
        CMS[i] = (SF * 0.01 * inputs["Area"] * 10000) / 86400
        #----------------------------------------------------------------------
    # return the result array	
    return CMS

# More ET method code can be found at https://github.com/phydrus/PyEt 
def cal_pet_Hamon(temp, Lat, start_date, dz=None):
    """Calculate potential evapotranspiration (pet) with Hamon (1961) equation.

    Args:
        temp (Array): [degC] Daily mean temperature.
        Lat (float): [deg] Latitude.
        start_date (str): yyyy/mm/dd.
        dz (float): [m] Altitude temperature adjustment. Defaults to None.

    Returns:
        [Array]: [cm/day] pet
    """
    temp = np.array(temp)
    # Altitude temperature adjustment
    if dz is not None:
        # Assume temperature decrease 0.6 degC for every 100 m elevation.
        tlaps = 0.6
        temp = temp - tlaps*dz/100
    # Calculate Julian days
    data_length = len(temp)
    start_date = to_datetime(start_date, format="%Y/%m/%d")                         
    pdDatedateIndex = date_range(start=start_date, periods=data_length,
                                 freq="D")
    JDay = to_numeric(pdDatedateIndex.strftime('%j')) # convert to Julian days
    # Calculate solar declination [rad] from day of year (JDay) based on
    # equations 24 in ALLen et al (1998).
    sol_dec = 0.4093 * np.sin(2. * 3.141592654 / 365. * JDay - 1.39)   
    Lat_rad = Lat*np.pi/180
    # Calculate sunset hour angle from latitude and solar declination [rad]
    # based on equations 25 in ALLen et al (1998).
    omega = np.arccos(-np.tan(sol_dec) * np.tan(Lat_rad))
    # Calculate maximum possible daylight length [hr]
    dl = 24 / np.pi * omega  
    # From Prudhomme(hess, 2013)
    # https://hess.copernicus.org/articles/17/1365/2013/hess-17-1365-2013-supplement.pdf
    # Slightly different from what we used to.
    pet = (dl / 12) ** 2 * np.exp(temp / 16) 
    pet = np.array(pet/10)         # Convert from mm to cm
    pet[np.where(temp <= 0)] = 0   # Force pet = 0 when temperature is below 0.
    return pet      # [cm/day]

#%%
r"""
# HYMOD model
Weather:
    T:                # [degC] Daily mean temperature.
    P:                # [cm] Daily precipitation.
    PE:               # [cm] Daily potential evapotranspiration.
Inputs:
    Area:             # [ha] Sub-basin area.
    SnowS:  5         # [cm] Snow storage.
    Latitude:         # [Degree] PET by Hamon.
    #s:               # [cm] Initial soil moisture.
    #Slow: 0          # [cm] Initial slow tank soil moisture.
    #Fast: [0,0,0]    # [cm] Initial fast tanks soil moisture.
HYMODPars:            ## For HYMOD, we have 6 parameters.
    Cmax:             # [cm] Maximum storage capacity. [1, 10]
    B:                # Degree of spatial variability of the soil moisture
                      ##capacity. [0, 2]
    Alpha:            # Factor distributing the flow between slow and quick
                      ##release reservoirs. [0.2, 0.99]
    Kq:               # Residence time of the slow release reservoir.
                      ##[0.5, 1.2]
    Ks:               # Residence time of the quick release reservoirs.
                      ##[0.01, 0.5]
    Df:               # [cm] Snow storage.
"""

def runHYMOD(HYMODPars, inputs, temp, prec, pet, data_length):
    """HYMOD for rainfall runoff simulation with additional snow module.
        Paper: 
        https://piahs.copernicus.org/articles/368/180/2015/piahs-368-180-2015.pdf
        Code: 
        https://github.com/bartnijssen/pythonlib/blob/master/hymod.py
    Args:
        HYMODPars (dict): [description]
        inputs (dict): [description]
        prec (Array): [cm] Daily precipitation.
        temp (Array): [degC] Daily mean temperature.
        pet (Array): [cm] Daily potential evaportranspiration.
        data_length (int): Total data length.
        
    Returns:
        [Array]: [cms] CMS (Qt)
    """
    CMS = np.zeros(data_length)   # Create a 1D array to store results
    Cmax = HYMODPars["Cmax"]*10   # [cm to mm] Upper limit of ET resistance
                                  ##parameter [0.5, 150]*10
    B = HYMODPars["B"]            # Distribution function shape parameter
                                  ##[0.01,4]
    Alpha = HYMODPars["Alpha"]    # Quick-slow split parameter [0.01,0.99]
    Kq = HYMODPars["Kq"]          # [1/day] Quick flow routing tanks rate
                                  ##parameter [0.01,0.99]
    Ks = HYMODPars["Ks"]          # [1/day] Slow flow routing tanks rate
                                  ##parameter [0.001,0.01]

    Df = HYMODPars["Df"]          # Snow melt coef.
    SnowSt = inputs["SnowS"]*10   # [cm to mm] Initial snow storage.

    prec = np.array(prec)*10      # [cm to mm] Daily precipitation.
    temp = np.array(temp)         # [degC] Daily mean temperature.
    pet = np.array(pet)*10        # [cm to mm] Daily potential ET.

    Sstar = 0 #inputs["InitS"]*10 # [cm to mm] Initial soil moisture
    Smax = Cmax / (1. + B)

    Ss = 0                        # Initial slow tank soil moisture.
    S1 = 0                
    S2 = 0
    S3 = 0

    for t in range(data_length): 
        # Snow module
        # Determine rainfall, snowfall and snow accumulation
        if temp[t] > 0:           # If temperature is above 0 degC, 
            # precipitation is rainfall (cm) and no snow accumulation
            Rt = prec[t]
        else:
            Rt = 0      # Else, precipitation is snowfall (cm) so rainfall = 0
            # Snowfall will accumulated and become snow storage(cm)
            SnowSt = SnowSt + prec[t]	
        # Determine snowmelt (Degree-day method)
        if temp[t] > 0:           # Temperature above 0 degC
            # Snowmelt (cm) capped by snow storage
            Mt = min(SnowSt, Df * temp[t])   
            SnowSt = SnowSt - Mt  # Update snow storage
        else:	
            Mt = 0
        P = Rt + Mt
        
        C = Cmax*(  1 - (1-((B+1)*Sstar)/(Cmax))**(1/(B+1)) )
        ER1 = max(P+C-Cmax,0)
        Cstar = min(P+C,Cmax)
        S = (Cmax/(B+1)) * (1-(1-(Cstar/Cmax))**(B+1))
        e = min(pet[t]*Cstar/Cmax,S)
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
        CMS[t] = (Q * 0.001 * inputs["Area"] * 10000) / 86400
        if (S<0 | Ss<0 | S1<0 | S2<0 | S3<0 | C<0 | Cstar<0 | e<0 | ER1<0
            | ER2<0 | Qs<0 | Qq1<0 | Qq2<0 | Qq3<0):
            print('infeasible')
            break
        return CMS


r"""
def runHYMOD(HYMODPars, inputs, temp, prec, pet, data_length):
    HYMOD for rainfall runoff simulation with additional snow module.
        Paper: https://piahs.copernicus.org/articles/368/180/2015/piahs-368-180-2015.pdf
        Code: https://github.com/bartnijssen/pythonlib/blob/master/hymod.py
    Args:
        HYMODPars (dict): [description]
        inputs (dict): [description]
        prec (Array): [cm] Daily precipitation.
        temp (Array): [degC] Daily mean temperature.
        pet (Array): [cm] Daily potential evaportranspiration.
        data_length (int): Total data length.
        
    Returns:
        [Array]: [cms] CMS (Qt)
    
    Cmax = HYMODPars["Cmax"]*10       # [cm to mm] Upper limit of ET
                                      ##resistance parameter
    Bexp = HYMODPars["Bexp"]          # Distribution function shape parameter
    Alpha = HYMODPars["Alpha"]        # Quick-slow split parameter
    Kq = HYMODPars["Kq"]              # Quick flow routing tanks rate parameter
    Ks = HYMODPars["Ks"]              # Slow flow routing tanks rate parameter
    Df = HYMODPars["Df"]              # Snow melt coef.
    SnowSt = inputs["SnowS"]*10       # [cm to mm] Initial snow storage.
    
    prec = np.array(prec)*10          # [cm to mm] Daily precipitation.
    temp = np.array(temp)             # [degC] Daily mean temperature.
    pet = np.array(pet)*10            # [cm to mm] Daily potential ET.
    
    s = inputs["s"]*10                  # cm to mm
    Smax = Cmax / (1. + Bexp)
    error = 0
    # Initialize slow tank state
    # value of 0 init flow works ok if calibration data starts with low
    # discharge
    x_slow = inputs["Slow"]*10              # cm to mm
    
    # Initialize state(s) of quick tank(s)
    x_quick = np.array(inputs["Fast"])*10   # cm to mm
    CMS = np.zeros(data_length) # Create a 1D array to store results
    
    #----- START PROGRAMMING LOOP WITH DETERMINING RAINFALL - RUNOFF AMOUNTS
    for i in range(data_length): 
        
        # Snow module
        # Determine rainfall, snowfall and snow accumulation
        if temp[t] > 0:           # If temperature is above 0 degC, 
            # precipitation is rainfall (cm) and no snow accumulation
            Rt = prec[t]
        else:
            Rt = 0      # Else, precipitation is snowfall (cm) so rainfall = 0
            # Snowfall will accumulated and become snow storage(cm)
            SnowSt = SnowSt + prec[t]	
        # Determine snowmelt (Degree-day method)
        if temp[t] > 0:           # Temperature above 0 degC
            # Snowmelt (cm) capped by snow storage
            Mt = min(SnowSt, Df * temp[t])   
            SnowSt = SnowSt - Mt  # Update snow storage
        else:	
            Mt = 0
        
        
        # Compute excess precipitation and evaporation
        ##ER1, ER2, x_loss = calExcess(x_loss, Cmax, Bexp, Rt+Mt, pet[i])
        
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
        # actual ET is linearly related to the soil moisture state
        evap = np.minimum(s1, s1/Smax * pet[i]) 
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
        CMS[i] = (Q * 0.001 * inputs["Area"] * 10000) / 86400
    return CMS
"""
