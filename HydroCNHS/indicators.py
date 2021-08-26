import numpy as np 
import pandas as pd

class Indicator(object):
    """
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
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def cal_indicator_df(x_obv, y_sim, indicators_list=None):
        dict = {"r"   : Indicator.r(x_obv, y_sim),
                "r2"  : Indicator.r2(x_obv, y_sim),
                "rmse": Indicator.rmse(x_obv, y_sim),
                "NSE" : Indicator.NSE(x_obv, y_sim),
                "iNSE": Indicator.iNSE(x_obv, y_sim),
                "KGE" : Indicator.KGE(x_obv, y_sim),
                "iKGE": Indicator.iKGE(x_obv, y_sim),
                "CP"  : Indicator.CP(x_obv, y_sim),
                "RSR" : Indicator.RSR(x_obv, y_sim)}
        df = pd.DataFrame(dict, index=["Value"])
        if indicators_list is None:
            return df
        else:
            return df.loc[:, indicators_list]
    
    @staticmethod
    def r(x_obv, y_sim):
        """Correlation of correlation

        Args:
            x_obv (Array): x or obv
            y_sim (Array): y or sim

        Returns:
            float
        """
        r = np.corrcoef(x_obv, y_sim)[0,1]
        if np.isnan(r):
            # We don't consider 2 identical horizontal line as r = 1!
            r = 0
        return r
    
    @staticmethod
    def r2(x_obv, y_sim):
        """Coefficient of determination

        Args:
            x_obv (Array): x or obv
            y_sim (Array): y or sim

        Returns:
            float
        """
        r = Indicator.r(x_obv, y_sim)
        return r**2
    
    @staticmethod
    def rmse(x_obv, y_sim):
        """Root mean square error

        Args:
            x_obv (Array): x or obv
            y_sim (Array): y or sim

        Returns:
            float
        """
        return np.nanmean((x_obv - y_sim)**2)**0.5
    
    @staticmethod
    def NSE(x_obv, y_sim):
        """Nash–Sutcliffe efficiency

        Args:
            x_obv (Array): x or obv
            y_sim (Array): y or sim

        Returns:
            float
        """
        mu_xObv = np.nanmean(x_obv)
        return 1 - np.nansum((x_obv-y_sim)**2) / np.nansum((x_obv-mu_xObv)**2)
    
    @staticmethod
    def iNSE(x_obv, y_sim):
        """Nash–Sutcliffe efficiency

        Args:
            x_obv (Array): x or obv
            y_sim (Array): y or sim

        Returns:
            float
        """
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
        return 1 - np.nansum((x_obv-y_sim)**2) / np.nansum((x_obv-mu_xObv)**2)
    
    @staticmethod
    def CP(x_obv, y_sim):
        """Correlation of persistence
        
        Args:
            x_obv (Array): x or obv
            y_sim (Array): y or sim

        Returns:
            float
        """
        a = np.nansum((x_obv[1:] - x_obv[:-1])**2)
        if a == 0:
            a = 0.0000001
        return 1 - np.nansum((x_obv[1:] - y_sim[1:])**2) / a
    
    @staticmethod
    def RSR(x_obv, y_sim):
        """RMSE-observations standard deviation ratio 

        Args:
            x_obv (Array): x or obv
            y_sim (Array): y or sim

        Returns:
            float
        """
        sig_xObv = np.nanstd(x_obv)
        return Indicator.rmse(x_obv, y_sim) / sig_xObv
    
    @staticmethod
    def KGE(x_obv, y_sim):
        """Kling–Gupta efficiency

        Args:
            x_obv (Array): x or obv
            y_sim (Array): y or sim

        Returns:
            float
        """
        mu_ySim = np.nanmean(y_sim); mu_xObv = np.nanmean(x_obv)
        sig_ySim = np.nanstd(y_sim); sig_xObv = np.nanstd(x_obv)
        kge = 1 - ((Indicator.r(x_obv, y_sim) - 1)**2 
                    + (sig_ySim/sig_xObv - 1)**2
                    + (mu_ySim/mu_xObv - 1)**2)**0.5
        return kge
    
    @staticmethod
    def iKGE(x_obv, y_sim):
        """Kling–Gupta efficiency with inverse transformed flow.
            https://www.fs.fed.us/nrs/pubs/jrnl/2015/nrs_2015_thirel_001.pdf

        Args:
            x_obv (Array): x or obv
            y_sim (Array): y or sim

        Returns:
            float
        """
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
        ikge = 1 - ((Indicator.r(x_obv, y_sim) - 1)**2
                    + (sig_ySim/sig_xObv - 1)**2
                    + (mu_ySim/mu_xObv - 1)**2)**0.5
        return ikge