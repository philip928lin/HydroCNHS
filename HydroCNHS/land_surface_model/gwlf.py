import numpy as np
from pandas import date_range, to_datetime, DataFrame

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