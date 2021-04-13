import numpy as np 

class Indicator():
    """
    r   : Correlation of correlation
    r2  : Coefficient of determination
    rmse: Root mean square error
    NSE : Nash–Sutcliffe efficiency
    CP  : Correlation of persistence
    RSR : RMSE-observations standard deviation ratio 
    KGE : Kling–Gupta efficiency
    iKGE: Inverse Kling–Gupta efficiency
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
        """Kling–Gupta efficiency

        Args:
            xObv (Array): x or obv
            ySim (Array): y or sim

        Returns:
            float
        """
        # Prevent dividing zero.
        xObv[xObv == 0] = 0.0001
        ySim[ySim == 0] = 0.0001
        xObv = 1/xObv
        ySim = 1/ySim
        mu_ySim = np.nanmean(ySim); mu_xObv = np.nanmean(xObv)
        sig_ySim = np.nanstd(ySim); sig_xObv = np.nanstd(xObv)
        return 1 - ((Indicator.r(xObv, ySim)-1)**2 + (sig_ySim/sig_xObv - 1)**2 + (mu_ySim/mu_xObv - 1)**2)**0.5