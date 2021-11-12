import numpy as np
import logging

# The ABCD model is mainly follow (Guillermo et al., 2010); however, with 
# different snow module. 
# https://doi.org/10.1029/2009WR008294
def run_ABCD(ABCD_pars, inputs, temp, prec, pet, data_length):
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
    a = ABCD_pars["a"]                   # [0, 1] ?
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