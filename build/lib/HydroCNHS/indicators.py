# Indicator module
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# Last update at 2021/12/23.

import numpy as np 
import pandas as pd

class Indicator(object):
    def __init__(self) -> None:
        """A class containing following indicator functions.
        
        r   : Correlation of correlation
        r2  : Coefficient of determination
        rmse: Root mean square error
        NSE : Nash–Sutcliffe efficiency
        iNSE: NSE with inverse transformed Q.
        CP  : Correlation of persistence
        RSR : RMSE-observations standard deviation ratio 
        KGE : Kling–Gupta efficiency
        iKGE: KGE with inverse transformed Q.
        """
        pass
    
    @staticmethod
    def remove_na(x_obv, y_sim):
        """Remove nan in x_obv and y_sim.
        
        This function makes sure there is no nan involves in the indicator
        calculation. If nan is detected, data points will be remove from x_obv 
        and y_sim simultaneously.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.

        Returns
        -------
        tuple
            Updated (x_obv, y_sim)
        """
        x_obv = np.array(x_obv)
        y_sim = np.array(y_sim)
        index = [True if np.isnan(x) == False and np.isnan(y) == False \
                    else False for x, y in zip(x_obv, y_sim)]
        x_obv = x_obv[index]
        y_sim = y_sim[index]
        print("Usable data ratio = {}/{}.".format(len(index), len(x_obv)))
        return x_obv, y_sim
    
    @staticmethod
    def cal_indicator_df(x_obv, y_sim, index_name="value",
                         indicators_list=None, r_na=True):
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
        dict = {"r"   : Indicator.r(x_obv, y_sim, False),
                "r2"  : Indicator.r2(x_obv, y_sim, False),
                "rmse": Indicator.rmse(x_obv, y_sim, False),
                "NSE" : Indicator.NSE(x_obv, y_sim, False),
                "iNSE": Indicator.iNSE(x_obv, y_sim, False),
                "KGE" : Indicator.KGE(x_obv, y_sim, False),
                "iKGE": Indicator.iKGE(x_obv, y_sim, False),
                "CP"  : Indicator.CP(x_obv, y_sim, False),
                "RSR" : Indicator.RSR(x_obv, y_sim, False)}
        df = pd.DataFrame(dict, index=[index_name])
        if indicators_list is None:
            return df
        else:
            return df.loc[:, indicators_list]
    
    @staticmethod
    def r(x_obv, y_sim, r_na=True):
        """Correlation.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        r_na : bool, optional
            Remove nan, by default True

        Returns
        -------
        float
            r coefficient.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
        r = np.corrcoef(x_obv, y_sim)[0,1]
        if np.isnan(r):
            # We don't consider 2 identical horizontal line as r = 1!
            r = 0
        return r
    
    @staticmethod
    def r2(x_obv, y_sim, r_na=True):
        """Coefficient of determination.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        r_na : bool, optional
            Remove nan, by default True

        Returns
        -------
        float
            r2 coefficient.
        """
        r = Indicator.r(x_obv, y_sim, r_na)
        return r**2
    
    @staticmethod
    def rmse(x_obv, y_sim, r_na=False):
        """Root mean square error.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        r_na : bool, optional
            Remove nan, by default True

        Returns
        -------
        float
            Root mean square error.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
        return np.nanmean((x_obv - y_sim)**2)**0.5
    
    @staticmethod
    def NSE(x_obv, y_sim, r_na=False):
        """Nash–Sutcliffe efficiency.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        r_na : bool, optional
            Remove nan, by default True

        Returns
        -------
        float
            Nash–Sutcliffe efficiency.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
        mu_xObv = np.nanmean(x_obv)
        return 1 - np.nansum((y_sim-x_obv)**2) / np.nansum((x_obv-mu_xObv)**2)
    
    @staticmethod
    def iNSE(x_obv, y_sim, r_na=False):
        """Inverse Nash–Sutcliffe efficiency.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        r_na : bool, optional
            Remove nan, by default True

        Returns
        -------
        float
            Inverse Nash–Sutcliffe efficiency.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
        # Prevent dividing zero.
        if np.nanmean(x_obv) == 0:
            x_obv = 1 / (x_obv + 0.0000001)
        else:
            x_obv = 1 / (x_obv + 0.01*np.nanmean(x_obv))
            
        if np.nanmean(y_sim) == 0:
            y_sim = 1 / (y_sim + 0.0000001)
        else:
            y_sim = 1 / (y_sim + 0.01*np.nanmean(y_sim))
        mu_xObv = np.nanmean(x_obv)
        return 1 - np.nansum((y_sim-x_obv)**2) / np.nansum((x_obv-mu_xObv)**2)
    
    @staticmethod
    def CP(x_obv, y_sim, r_na=False):
        """Correlation of persistence.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        r_na : bool, optional
            Remove nan, by default True

        Returns
        -------
        float
            Correlation of persistence.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
        a = np.nansum((x_obv[1:] - x_obv[:-1])**2)
        if a == 0:
            a = 0.0000001
        return 1 - np.nansum((x_obv[1:] - y_sim[1:])**2) / a
    
    @staticmethod
    def RSR(x_obv, y_sim, r_na=False):
        """RMSE-observations standard deviation ratio.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        r_na : bool, optional
            Remove nan, by default True

        Returns
        -------
        float
            RMSE-observations standard deviation ratio.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
        sig_xObv = np.nanstd(x_obv)
        return Indicator.rmse(x_obv, y_sim) / sig_xObv
    
    @staticmethod
    def KGE(x_obv, y_sim, r_na=True):
        """Kling–Gupta efficiency.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        r_na : bool, optional
            Remove nan, by default True

        Returns
        -------
        float
            Kling–Gupta efficiency.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
            
        mu_ySim = np.nanmean(y_sim); mu_xObv = np.nanmean(x_obv)
        sig_ySim = np.nanstd(y_sim); sig_xObv = np.nanstd(x_obv)
        kge = 1 - ((Indicator.r(x_obv, y_sim, False) - 1)**2 
                    + (sig_ySim/sig_xObv - 1)**2
                    + (mu_ySim/mu_xObv - 1)**2)**0.5
        return kge
    
    @staticmethod
    def iKGE(x_obv, y_sim, r_na=True):
        """Inverse Kling–Gupta efficiency.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        r_na : bool, optional
            Remove nan, by default True

        Returns
        -------
        float
            Inverse Kling–Gupta efficiency.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
            
        # Prevent dividing zero.
        if np.nanmean(x_obv) == 0:
            x_obv = 1/(x_obv + 0.0000001)
        else:
            x_obv = 1/(x_obv + 0.01*np.nanmean(x_obv))
            
        if np.nanmean(y_sim) == 0:
            y_sim = 1/(y_sim + 0.0000001)
        else:
            y_sim = 1/(y_sim + 0.01*np.nanmean(y_sim))
            
        mu_ySim = np.nanmean(y_sim); mu_xObv = np.nanmean(x_obv)
        sig_ySim = np.nanstd(y_sim); sig_xObv = np.nanstd(x_obv)
        ikge = 1 - ((Indicator.r(x_obv, y_sim, False) - 1)**2
                    + (sig_ySim/sig_xObv - 1)**2
                    + (mu_ySim/mu_xObv - 1)**2)**0.5
        return ikge