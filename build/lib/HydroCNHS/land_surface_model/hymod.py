import numpy as np

# HYMOD model
r"""
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

def run_HYMOD(HYMODPars, inputs, temp, prec, pet, data_length):
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
