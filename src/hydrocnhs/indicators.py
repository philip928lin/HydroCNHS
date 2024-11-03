# Indicator module
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com)
# Last update at 2021/12/23.

import numpy as np
import pandas as pd
import warnings

class Indicator:
    """A class for calculating indicators."""

    def __init__(self) -> None:
        """Initialize an Indicator class to manage various hydrological calculations.

        r   : Correlation of correlation
        r2  : Coefficient of determination
        rmse: Root mean square error
        nse : Nash-Sutcliffe efficiency
        inse: nse with inverse transformed Q.
        cp  : Correlation of persistence
        rsr : RMSE-observations standard deviation ratio
        kge : Kling-Gupta efficiency
        ikge: kge with inverse transformed Q.

        Note
        ----
        The code is adopted from HydroCNHS (Lin et al., 2022).
        Lin, C. Y., Yang, Y. C. E., & Wi, S. (2022). HydroCNHS: A Python Package of
        Hydrological Model for Coupled Natural-Human Systems. Journal of Water
        Resources Planning and Management, 148(12), 06022005.
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
    
        # Create a mask where neither x_obv nor y_sim has NaN
        mask = ~np.isnan(x_obv) & ~np.isnan(y_sim)
    
        # Apply the mask to both arrays
        x_obv = x_obv[mask]
        y_sim = y_sim[mask]
    
        return x_obv, y_sim

    @staticmethod
    def cal_indicator_df(
        x_obv, y_sim, index_name="value", indicators_list=None, r_na=True
    ):
        """Calculate indicators and return as a DataFrame.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        index_name : str, optional
            Index name, by default "value".
        indicators_list : list, optional
            List of indicators, by default None.
        r_na : bool, optional
            Remove nan, by default True.

        Returns
        -------
        DataFrame
            A DataFrame of indicators.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            indicators_dict = {
                "r": Indicator.get_r(x_obv, y_sim, False),
                "r2": Indicator.get_r2(x_obv, y_sim, False),
                "rmse": Indicator.get_rmse(x_obv, y_sim, False),
                "nse": Indicator.get_nse(x_obv, y_sim, False),
                "inse": Indicator.get_inse(x_obv, y_sim, False),
                "kge": Indicator.get_kge(x_obv, y_sim, False),
                "ikge": Indicator.get_ikge(x_obv, y_sim, False),
                "cp": Indicator.get_cp(x_obv, y_sim, False),
                "rsr": Indicator.get_rsr(x_obv, y_sim, False),
            }
            df = pd.DataFrame(indicators_dict, index=[index_name])
            if indicators_list is None:
                return df
            else:
                return df.loc[:, indicators_list]
      
    @staticmethod
    def get_df_indicator(x_obv_df, y_sim_df, indicators_list=None, r_na=True):
        """Calculate indicators and return as a DataFrame.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        indicators_list : list, optional
            List of indicators, by default None.
        r_na : bool, optional
            Remove nan, by default True.

        Returns
        -------
        DataFrame
            A DataFrame of indicators.
        """
        # Get common columns between two dataframes
        common_columns = x_obv_df.columns.intersection(y_sim_df.columns)
        
        results = pd.DataFrame(index=indicators_list)
        for col in common_columns:
            x_obv = x_obv_df[col]
            y_sim = y_sim_df[col]
            
            if r_na:
                x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
    
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                
                inds = [eval(f"Indicator.get_{ind}(x_obv, y_sim, False)") for ind in indicators_list]
                results[col] = inds

        return results
    
    @staticmethod
    def get_r(x_obv, y_sim, r_na=True):
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
        r = np.corrcoef(x_obv, y_sim)[0, 1]
        if np.isnan(r):
            # We don't consider 2 identical horizontal line as r = 1!
            r = 0
        return r

    @staticmethod
    def get_r2(x_obv, y_sim, r_na=True):
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
        r = Indicator.get_r(x_obv, y_sim, r_na)
        return r**2

    @staticmethod
    def get_rmse(x_obv, y_sim, r_na=False):
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
        return np.nanmean((x_obv - y_sim) ** 2) ** 0.5

    @staticmethod
    def get_nse(x_obv, y_sim, r_na=False):
        """Nash-Sutcliffe efficiency.

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
            Nash-Sutcliffe efficiency.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)

        mu_xObv = np.nanmean(x_obv)
        return 1 - np.nansum((y_sim - x_obv) ** 2) / np.nansum((x_obv - mu_xObv) ** 2)

    @staticmethod
    def get_inse(x_obv, y_sim, r_na=False):
        """Inverse Nash-Sutcliffe efficiency.

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
            Inverse Nash-Sutcliffe efficiency.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
        # Prevent dividing zero.
        if np.nanmean(x_obv) == 0:
            x_obv = 1 / (x_obv + 0.0000001)
        else:
            x_obv = 1 / (x_obv + 0.01 * np.nanmean(x_obv))

        if np.nanmean(y_sim) == 0:
            y_sim = 1 / (y_sim + 0.0000001)
        else:
            y_sim = 1 / (y_sim + 0.01 * np.nanmean(y_sim))

        mu_xObv = np.nanmean(x_obv)
        return 1 - np.nansum((y_sim - x_obv) ** 2) / np.nansum((x_obv - mu_xObv) ** 2)

    @staticmethod
    def get_cp(x_obv, y_sim, r_na=False):
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
        a = np.nansum((x_obv[1:] - x_obv[:-1]) ** 2)
        if a == 0:
            a = 0.0000001
        return 1 - np.nansum((x_obv[1:] - y_sim[1:]) ** 2) / a

    @staticmethod
    def get_rsr(x_obv, y_sim, r_na=False):
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
        return Indicator.get_rmse(x_obv, y_sim) / sig_xObv

    @staticmethod
    def get_kge(x_obv, y_sim, r_na=True):
        """Kling-Gupta efficiency.

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
            Kling-Gupta efficiency.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)

        mu_ySim = np.nanmean(y_sim)
        mu_xObv = np.nanmean(x_obv)
        sig_ySim = np.nanstd(y_sim)
        sig_xObv = np.nanstd(x_obv)
        kge = (
            1
            - (
                (Indicator.get_r(x_obv, y_sim, False) - 1) ** 2
                + (sig_ySim / sig_xObv - 1) ** 2
                + (mu_ySim / mu_xObv - 1) ** 2
            )
            ** 0.5
        )
        return kge

    @staticmethod
    def get_ikge(x_obv, y_sim, r_na=True):
        """Inverse Kling-Gupta efficiency.

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
            Inverse Kling-Gupta efficiency.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)

        # Prevent dividing zero.
        if np.nanmean(x_obv) == 0:
            x_obv = 1 / (x_obv + 0.0000001)
        else:
            x_obv = 1 / (x_obv + 0.01 * np.nanmean(x_obv))

        if np.nanmean(y_sim) == 0:
            y_sim = 1 / (y_sim + 0.0000001)
        else:
            y_sim = 1 / (y_sim + 0.01 * np.nanmean(y_sim))

        mu_ySim = np.nanmean(y_sim)
        mu_xObv = np.nanmean(x_obv)
        sig_ySim = np.nanstd(y_sim)
        sig_xObv = np.nanstd(x_obv)
        ikge = (
            1
            - (
                (Indicator.get_r(x_obv, y_sim, False) - 1) ** 2
                + (sig_ySim / sig_xObv - 1) ** 2
                + (mu_ySim / mu_xObv - 1) ** 2
            )
            ** 0.5
        )
        return ikge
