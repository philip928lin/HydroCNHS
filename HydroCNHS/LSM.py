#%%
# Land Surface model using GWLF model.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# GWLF is based on the code from Ethan Yang @ Lehigh University (yey217@lehigh.edu)
# 2021/02/05
import numpy as np
from pandas import date_range, to_datetime, to_numeric
#import logging
#logger = logging.getLogger("HydroCNHS.HP") # Get logger for logging msg.

r"""
Weather:
    T:                                        # [degC] Daily mean temperature.
    P:                                        # [cm] Daily precipitation.
    PE:                                       # [cm] Daily potential evapotranspiration.
Inputs:
    Area:                                     # [ha] Sub-basin area.
    S0:     10                                # [cm] Shallow saturated soil water content.
    U0:     10                                # [cm] Unsaturated soil water content
    SnowS:  5                                 # [cm] Snow storage.
    MonthlyTavg: []                           # [degC] Monthly mean temperature.
GWLFPars:                                       # For GWLF, we have 9 parameters.
    CN2:                                      # Curve number
    IS:                                       # Interception coefficient 0.05 ~ 0.2
    Res:                                      # Recession coefficient
    Sep:                                      # Deep seepage coefficient
    Alpha:                                    # Baseflow coefficient (Eq 4 in Luo et al. (2012))
    Beta:                                     # Deep seepage coefficient (Eq 2 in Luo et al. (2012))
    Ur:                                       # [cm] Avaliable/Soil water capacity (Root zone)
    Df:                                       # [cm/degC] Degree-day coefficient for snowmelt.
    Kc:                                       # Land cover coefficient.
"""

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
        [Array]: [cms] Qt
    """
    #----- Setup initial values -----------------------------------------------------------
    Gt =  GWLFPars["Res"]*Inputs["S0"]  # [cm] Initialize saturated zone discharge to the stream.
    BFt = Gt						    # [cm] Initialize baseflow to stream.
    St = Inputs["S0"]                   # [cm] Initialize shallow saturated soil water content.
    Ut = Inputs["U0"]                   # [cm] Initialize unsaturated soil water content.
    AnteMois = [0, 0, 0, 0, 0]          # [cm] Define the initial Antecedent Moisture (5 days) as 0.
    MonthlyTavg = np.array(Inputs["MonthlyTavg"]) # Monthly mean temperature.
    Tt = np.array(Tt)                   # [degC] Daily mean temperature.
    Pt = np.array(Pt)                   # [cm] Daily precipitation.
    PEt = np.array(PEt)                 # [cm] Daily potential evapotranspiration.
    
    # Calculate month index for each data point for realizing growing season purpose.
    StartDate = to_datetime(StartDate, format="%Y/%m/%d")                               # to Datetime
    pdDatedateIndex = date_range(start = StartDate, periods = DataLength, freq = "D")   # gen pd dateIndex
    m = to_numeric(pdDatedateIndex.strftime('%m'))                                      # get month
    m = np.array(m)                                                                     # to array
    #--------------------------------------------------------------------------------------

    #----- Loop through all days (Python for loop ending needs +1) ------------------------
    CMS = np.zeros(DataLength) # Create a 1D array to store results
    for i in range(DataLength): 	
        # Determine the am1 and am2 values based on growing season for CN calculation.
        # Growing season (mean monthly temperature > 10 deg C).
        if MonthlyTavg[int(m[i])-1]> 10: # "m" = [1~12] to fit index 0 (=Jan) to 11 (=Dec)
            am1 = 3.6
            am2 = 5.3
        else:
            am1 = 1.3
            am2 = 2.8
            
        # Determine rainfall, snowfall and snow accumulation-------------------------------
        if Tt[i] > 0:           # If temperature is above 0 degC, 
            Rt = Pt[i]          # precipitation is rainfall (cm) and no snow accumulation
        else:
            Rt = 0              # Else, precipitation is snowfall (cm) so rainfall = 0
            Inputs["SnowS"] = Inputs["SnowS"] + Pt[i] # Snowfall will accumulated and become snow storage(cm)
        #---------------------------------------------------------------------------------------------		
        # Determine snowmelt (Degree-day method)------------------------------------------------------
        if Tt[i] > 0:           # Temperature above 0 degC
            Mt = min(Inputs["SnowS"], GWLFPars["Df"] * Tt[i])   # Snowmelt (cm) capped by snow storage
            Inputs["SnowS"] = Inputs["SnowS"] - Mt              # Update snow storage
        else:	
            Mt = 0
        #---------------------------------------------------------------------------------------------	
        # Calculate the Antecedent Moisture (at)------------------------------------------------------
        AnteMois = [Rt + Mt] + AnteMois[0:4]    # Add new data and drop last data.
        at = np.sum(AnteMois)                   # Five days antecedent moisture
        #---------------------------------------------------------------------------------------------
        # CN calculation (Following GWLF2 setting)----------------------------------------------------
        CN1 = (4.2 * GWLFPars["CN2"]) / (10 - 0.058 * GWLFPars["CN2"])
        CN3 = (23 * GWLFPars["CN2"]) / (10 + 0.13 * GWLFPars["CN2"])
        CN = None
        if at < am1:
            CN = CN1 + ((GWLFPars["CN2"] - CN1) / am1) * at
            
        if (am1 < at and at < am2):
            CN = GWLFPars["CN2"] + ((CN3 - GWLFPars["CN2"]) / (am2 - am1)) * (at - am1)       
            
        if am2 < at: 
            CN = CN3	
        #---------------------------------------------------------------------------------------------
        # Calculate runoff (Qt)-----------------------------------------------------------------------
        DSkt = (2540 / CN) - 25.4               # Detention parameter (cm)
        if (Rt + Mt) > GWLFPars["IS"] * DSkt:
            Qt = (((Rt + Mt) - (GWLFPars["IS"] * DSkt))**2) / ((Rt + Mt) + ((1-GWLFPars["IS"]) * DSkt))
        else:
            Qt = 0	
        #---------------------------------------------------------------------------------------------
        # Calculate Evapotranspiration (Et)-----------------------------------------------------------
        # Consider water stress (Ks) and land cover (Kc).
        if (Ut >= GWLFPars["Ur"]*0.5):
            Ks = 1
        else:
            Ks = Ut/(GWLFPars["Ur"]*0.5)
        Et = min((Ut+ Rt + Mt - Qt), Ks*GWLFPars["Kc"]*PEt[i])
        #---------------------------------------------------------------------------------------------
        # Calculate Percolation (Pct) from unsaturated zone (Ut) to shallow saturated zone (St)------- 
        PCt = max((Ut+ Rt + Mt - Qt - Et - GWLFPars["Ur"]), 0)
        #---------------------------------------------------------------------------------------------
        # Update unsaturated zone soil moistures (Ut)-------------------------------------------------
        Ut = Ut + Rt + Mt - Qt - Et - PCt       # Rt+Mt-Qt is infiltration
        #---------------------------------------------------------------------------------------------
        # Calculate groundwater discharge (Gt) and deep seepage (Dt)----------------------------------  
        Gt = GWLFPars["Res"] * St
        Dt = GWLFPars["Sep"] * St
        #---------------------------------------------------------------------------------------------
        # Update shallow saturated zone soil moistures (St)-------------------------------------------
        St = St + PCt - Gt - Dt
        #---------------------------------------------------------------------------------------------
        # Groundwater: Deep seepage loss (Dset)-------------------------------------------------------
        Dset = GWLFPars["Beta"] * Dt #Eq 2 in Luo et al. (2012)
        #---------------------------------------------------------------------------------------------
        # Groundwater: Recharge (Ret)-----------------------------------------------------------------
        Ret = Dt - Dset #Eq 3 in Luo et al. (2012)
        #---------------------------------------------------------------------------------------------	
        # Groundwater:Baseflow (BFt)------------------------------------------------------------------
        BFt = BFt * np.exp(-GWLFPars["Alpha"]) + Ret * (1-np.exp(-GWLFPars["Alpha"])) #Eq 4 in Luo et al. (2012)
        #---------------------------------------------------------------------------------------------
        # Calculate streamflow (SF) and Monthly Qt, Pt, Et, Gt and SF---------------------------------
        SF = Qt + Gt + BFt #Streamflow = surface quick flow + subsurface flow + baseflow
        #---------------------------------------------------------------------------------------------			
        # Change unit to cms (m^3/sec)----------------------------------------------------------------
        CMS[i] = (SF * 0.01 * Inputs["Area"] * 10000) / 86400
        #---------------------------------------------------------------------------------------------
    # return the result array	
    #logger.info("[GWLF] Complete runoff simulation.")
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

#%% Test function
r"""
Inputs = {}
Inputs["Area"] = 5000
Inputs["S0"] = 10
Inputs["U0"] = 10
Inputs["SnowS"] = 5 
Inputs["MonthlyTavg"] = [-5.74, -4.35, 1.06, 7.73, 14.24, 19.37, 21.71, 20.60, 16.54, 10.19, 3.65, -2.75] 

GWLFPars = {}
GWLFPars["CN2"] = 33.18
GWLFPars["IS"] = 0.0527
GWLFPars["Res"] = 0.196
GWLFPars["Sep"] = 0.0975
GWLFPars["Alpha"] = 0.058
GWLFPars["Beta"] = 0.766
GWLFPars["Ur"] = 14.387
GWLFPars["Df"] = 0.176
GWLFPars["Kc"] = 1

Tt = np.array([0.598333333,-3.431666667,-0.888333333,2.29,4.785,3.48,1.618333333,0.285,-0.055,1.373333333,4.43333333,3.49,5.736666667,6.253333333,11.50666667,3.038333333,0.443333333,3.64,6.84,7.631666667,12.8666667,9.028333333,11.17833333,13.99333333,7.828333333,6.051666667,8.681666667,5.953333333,4.07,7.41666667,5.8,2.835,7.77,8.365,8.22,12.02833333,16.90833333,16.50833333,10.41333333,6.968333333,11.64,17.99666667,19.80333333,21.53833333,18.13833333,10.99,8.765,8.92,10.31333333,13.015,10.87666667,8.381666667,12.62333333,15.88333333,13.96166667,4.89,8.785,16.26666667,12.81,11.50333333,16.365])
Pt = np.array([0.598333333,-3.431666667,-0.888333333,2.29,4.785,3.48,1.618333333,0.285,-0.055,1.373333333,4.543333333,3.49,5.736666667,6.253333333,11.50666667,3.038333333,0.443333333,3.64,6.84,7.631666667,12.38666667,9.028333333,11.17833333,13.99333333,7.828333333,6.051666667,8.681666667,5.953333333,4.07,7.141666667,5.8,2.835,7.77,8.365,8.22,12.02833333,16.90833333,16.50833333,10.41333333,6.968333333,11.64,17.99666667,19.80333333,21.53833333,18.13833333,10.99,8.765,8.92,10.31333333,13.015,10.87666667,8.381666667,12.62333333,15.88333333,13.96166667,4.89,8.785,16.26666667,12.81,11.50333333,16.365])
StartDate = "1961/04/05"
PEt = calPEt_Hamon(Tt, Lat = 42.648, StartDate = StartDate, dz = None)
DataLength = 61

Qt = runGWLF(GWLFPars, Inputs, Tt, Pt, PEt, StartDate, DataLength)
"""