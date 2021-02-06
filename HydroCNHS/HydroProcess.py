#%%
# Land Surface model using GWLF model.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# based on the code from Ethan Yang @ Lehigh University (yey217@lehigh.edu)
# 2021/02/05
import numpy as np
import logging
logger = logging.getLogger("HydroCNHS.HP") # Get logger for logging msg.

"""
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


def GWLF(GWLFPars, Inputs, Tt, Pt, PEt, m, DataLength):
    """GWLF for rainfall runoff simulation.

    Args:
        GWLFPars (dict): Contain 9 parameters.  
        Inputs (dict): Contain 5 inputs
        Tt (Array): [degC] Daily mean temperature.
        Pt (Array): [cm] Daily precipitation.
        PEt (Array): [cm] Daily potential evaportranspiration.
        m (Array): Month of each data point. 
        DataLength (int): Total data length.
    """
    #----- Setup initial values -----------------------------------------------------------
    Gt =  GWLFPars["Res"]*Inputs["S0"]  # [cm] Initialize saturated zone discharge to the stream.
    BFt = Gt						    # [cm] Initialize baseflow to stream.
    St = Inputs["S0"]                   # [cm] Initialize shallow saturated soil water content.
    Ut = Inputs["U0"]                   # [cm] Initialize unsaturated soil water content.
    AnteMois = [0, 0, 0, 0, 0]          # [cm] Define the initial Antecedent Moisture (5 days) as 0.
    MonthlyTavg = Inputs["MonthlyTavg"] # Monthly mean temperature.
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
        Ut = Ut + Rt + Mt - Qt - Et - PCt       # Rt+Mt-Qt-Et can be seen as infiltration
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
    logger.info("[GWLF] Complete runoff simulation.")
    return CMS