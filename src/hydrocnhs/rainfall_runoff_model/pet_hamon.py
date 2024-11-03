# Hamon pet module.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com)
# Last update at 2021/12/22.

import numpy as np
from pandas import date_range, to_datetime, to_numeric


def cal_pet_Hamon(temp, Lat, start_date, dz=None):
    """Calculate potential evapotranspiration (pet) with Hamon (1961) equation.

    Parameters
    ----------
    temp : array
        [degC] Daily mean temperature.
    Lat : float
        [deg] Latitude.
    start_date : str
        Start date "yyyy/mm/dd".
    dz : float, optional
        [m] Altitude temperature adjustment, by default None.

    Returns
    -------
    array
        [cm/day] Potential evapotranspiration
    """
    temp = np.array(temp)
    # Altitude temperature adjustment
    if dz is not None:
        # Assume temperature decrease 0.6 degC for every 100 m elevation.
        tlaps = 0.6
        temp = temp - tlaps * dz / 100
    # Calculate Julian days
    if isinstance(start_date, str):
        data_length = len(temp)
        start_date = to_datetime(start_date, format="%Y/%m/%d")
        pdDatedateIndex = date_range(start=start_date, periods=data_length, freq="D")
    else:
        pdDatedateIndex = start_date  # for internal use.
    JDay = to_numeric(pdDatedateIndex.strftime("%j"))  # convert to Julian days
    # Calculate solar declination [rad] from day of year (JDay) based on
    # equations 24 in ALLen et al (1998).
    sol_dec = 0.4093 * np.sin(2.0 * 3.141592654 / 365.0 * JDay - 1.39)
    Lat_rad = Lat * np.pi / 180
    # Calculate sunset hour angle from latitude and solar declination [rad]
    # based on equations 25 in ALLen et al (1998).
    omega = np.arccos(-np.tan(sol_dec) * np.tan(Lat_rad))
    # Calculate maximum possible daylight length [hr]
    dl = 24 / np.pi * omega
    # From Prudhomme(hess, 2013)
    # https://hess.copernicus.org/articles/17/1365/2013/hess-17-1365-2013-supplement.pdf
    # Slightly different from what we used to.
    pet = (dl / 12) ** 2 * np.exp(temp / 16)
    pet = np.array(pet / 10)  # Convert from mm to cm
    pet[np.where(temp <= 0)] = 0  # Force pet = 0 when temperature is below 0.
    return pet  # [cm/day]


# More ET method code can be found at https://github.com/phydrus/PyEt
