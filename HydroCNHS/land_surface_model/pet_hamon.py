import numpy as np
from pandas import date_range, to_datetime, to_numeric

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