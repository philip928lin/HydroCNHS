import numpy as np 
import pandas as pd

class Indicator():
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
    # r = r
    # r2 = r2
    # rmse = rmse
    # NSE = NSE
    # CP = CP
    # RSR = RSR
    # KGE = KGE
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def calIndicatorDF(xObv, ySim, IndicatorsList = None):
        Dict = {"r"   : Indicator.r(xObv, ySim),
                "r2"  : Indicator.r2(xObv, ySim),
                "rmse": Indicator.rmse(xObv, ySim),
                "NSE" : Indicator.NSE(xObv, ySim),
                "iNSE": Indicator.iNSE(xObv, ySim),
                "KGE" : Indicator.KGE(xObv, ySim),
                "iKGE": Indicator.iKGE(xObv, ySim),
                "CP"  : Indicator.CP(xObv, ySim),
                "RSR" : Indicator.RSR(xObv, ySim)}
        DF = pd.DataFrame(Dict)
        if IndicatorsList is None:
            return DF
        else:
            return DF.loc[:, IndicatorsList]
    
    @staticmethod
    def r(xObv, ySim):
        """Correlation of correlation

        Args:
            xObv (Array): x or obv
            ySim (Array): y or sim

        Returns:
            float
        """
        return np.corrcoef(xObv, ySim)[0,1]
    
    @staticmethod
    def r2(xObv, ySim):
        """Coefficient of determination

        Args:
            xObv (Array): x or obv
            ySim (Array): y or sim

        Returns:
            float
        """
        return np.corrcoef(xObv, ySim)[0,1]**2
    
    @staticmethod
    def rmse(xObv, ySim):
        """Root mean square error

        Args:
            xObv (Array): x or obv
            ySim (Array): y or sim

        Returns:
            float
        """
        return np.nanmean((xObv-ySim)**2)**0.5
    
    @staticmethod
    def NSE(xObv, ySim):
        """Nash–Sutcliffe efficiency

        Args:
            xObv (Array): x or obv
            ySim (Array): y or sim

        Returns:
            float
        """
        mu_xObv = np.nanmean(xObv)
        return 1 - np.nansum((xObv-ySim)**2)/np.nansum((xObv-mu_xObv)**2) # Nash
    
    @staticmethod
    def iNSE(xObv, ySim):
        """Nash–Sutcliffe efficiency

        Args:
            xObv (Array): x or obv
            ySim (Array): y or sim

        Returns:
            float
        """
        # Prevent dividing zero.
        xObv = 1/(xObv + 0.01*np.nanmean(xObv))
        ySim = 1/(ySim + 0.01*np.nanmean(ySim))
        mu_ySim = np.nanmean(ySim); mu_xObv = np.nanmean(xObv)
        sig_ySim = np.nanstd(ySim); sig_xObv = np.nanstd(xObv)
        mu_xObv = np.nanmean(xObv)
        return 1 - np.nansum((xObv-ySim)**2)/np.nansum((xObv-mu_xObv)**2) # Nash
    
    @staticmethod
    def CP(xObv, ySim):
        """Correlation of persistence
        
        Args:
            xObv (Array): x or obv
            ySim (Array): y or sim

        Returns:
            float
        """
        return 1 - np.nansum((xObv[1:]-ySim[1:])**2)/np.nansum((xObv[1:]-xObv[:-1])**2)
    
    @staticmethod
    def RSR(xObv, ySim):
        """RMSE-observations standard deviation ratio 

        Args:
            xObv (Array): x or obv
            ySim (Array): y or sim

        Returns:
            float
        """
        sig_xObv = np.nanstd(xObv)
        return Indicator.rmse(xObv, ySim)/sig_xObv
    
    @staticmethod
    def KGE(xObv, ySim):
        """Kling–Gupta efficiency

        Args:
            xObv (Array): x or obv
            ySim (Array): y or sim

        Returns:
            float
        """
        mu_ySim = np.nanmean(ySim); mu_xObv = np.nanmean(xObv)
        sig_ySim = np.nanstd(ySim); sig_xObv = np.nanstd(xObv)
        return 1 - ((Indicator.r(xObv, ySim)-1)**2 + (sig_ySim/sig_xObv - 1)**2 + (mu_ySim/mu_xObv - 1)**2)**0.5
    
    @staticmethod
    def iKGE(xObv, ySim):
        """Kling–Gupta efficiency with inverse transformed flow.
            https://www.fs.fed.us/nrs/pubs/jrnl/2015/nrs_2015_thirel_001.pdf

        Args:
            xObv (Array): x or obv
            ySim (Array): y or sim

        Returns:
            float
        """
        # Prevent dividing zero.
        xObv = 1/(xObv + 0.01*np.nanmean(xObv))
        ySim = 1/(ySim + 0.01*np.nanmean(ySim))
        mu_ySim = np.nanmean(ySim); mu_xObv = np.nanmean(xObv)
        sig_ySim = np.nanstd(ySim); sig_xObv = np.nanstd(xObv)
        return 1 - ((Indicator.r(xObv, ySim)-1)**2 + (sig_ySim/sig_xObv - 1)**2 + (mu_ySim/mu_xObv - 1)**2)**0.5