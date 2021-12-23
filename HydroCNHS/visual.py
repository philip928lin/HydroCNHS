# Visualization module
# This module is not the core module, which is unlikely to be maintained. 
# Some simple functions for users to quickly visualize the simulated results.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# Last update at 2021/12/23.
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from .indicators import Indicator

class Visual():
    """Collection of some plotting functions.
    """
    @staticmethod
    def plot_reg(x_obv, y_sim, title=None, xy_labal=None, same_xy_limit=True,
                 return_reg_par=False, save_fig_path=None, show=True):
        """Plot regression.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        title : str, optional
            Title, by default None.
        xy_labal : list, optional
            List of x and y labels, by default None.
        same_xy_limit : bool, optional
            If True same limit will be applied to x and y axis, by default True.
        return_reg_par : bool, optional
            If True, slope and interception will be return, by default False.
        save_fig_path : str, optional
            If given, plot will be save as .png, by default None.
        show : bool, optional
            If True, the plot will be shown in the console, by default True.

        Returns
        -------
        ax or list
            axis object or [slope, intercept].
        """
        if title is None:
            title = "Regression" 
        else:
            title = title
        
        if xy_labal is None:
            x_label = "Obv"; y_label = "Sim"
        else:
            x_label = xy_labal[0]; y_label = xy_labal[1]
            
        # Create figure
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        # Regression calculation and plot
        x_obv = np.array(x_obv)
        y_sim = np.array(y_sim)
        mask = ~np.isnan(x_obv) & ~np.isnan(y_sim)  # Mask to ignore nan
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x_obv[mask], y_sim[mask]) # Calculate the regression line
        line = slope*x_obv+intercept  # For plotting regression line
        ax.plot(x_obv, line, 'r', label='y={:.2f}x+{:.2f}'.format(slope,
                                                                  intercept))

        # Plot data point
        ax.scatter(x_obv, y_sim, color="k", s=3.5)
        ax.legend(fontsize=9, loc='upper right')
        if same_xy_limit:
            Max = max([np.nanmax(x_obv), np.nanmax(y_sim)])
            Min = min([np.nanmin(x_obv), np.nanmin(y_sim)])
            ax.set_xlim(Min, Max)
            ax.set_ylim(Min, Max)
            # Add 45 degree line
            interval = (Max - Min) / 10
            diagonal = np.arange(Min, Max+interval, interval)
            ax.plot(diagonal, diagonal, "b", linestyle='dashed', lw=1)
        
        
        # PLot indicators
        name = {"r": "$r$",
                "r2":"$r^2$",
                "rmse":"RMSE",
                "NSE": "NSE",
                "CP": "CP",
                "RSR": "RSR",
                "KGE": "KGE"}
        indicators = {}
        indicators["r"] = Indicator.r(x_obv, y_sim)
        indicators["r2"] = Indicator.r2(x_obv, y_sim)
        indicators["rmse"] = Indicator.rmse(x_obv, y_sim)
        indicators["NSE"] = Indicator.NSE(x_obv, y_sim)
        indicators["KGE"] = Indicator.KGE(x_obv, y_sim)
        
        string = "\n".join(['{:^4}: {}'.format(name[keys], round(values,5))
                            for keys,values in indicators.items()])
        
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
        ax.annotate(string, xy=(0.05, 0.95), xycoords='axes fraction',
                    verticalalignment='top', horizontalalignment='left',
                    transform=ax.transAxes, fontsize=9, bbox=props)       
        if show:
            plt.show()
        
        if save_fig_path is not None:
            fig.savefig(save_fig_path)
            plt.close()
            
        if return_reg_par:
            return [slope, intercept]
        else:
            return ax

    @staticmethod
    def plot_timeseries(x_obv, y_sim, xticks=None, title=None, xy_labal=None,
                       save_fig_path=None, legend=True, show=True, **kwargs):   
        """Plot timeseries.
        
        This function can plot two DataFrames with same column names.

        Parameters
        ----------
        x_obv : array/DataFrame
            Observation data.
        y_sim : array/DataFrame
            Simulation data.
        xticks : list, optional
            Ticks for x-axis, by default None.
        title : str, optional
            Title, by default None.
        xy_labal : list, optional
            List of x and y labels, by default None.
        save_fig_path : str, optional
            If given, plot will be save as .png, by default None.
        legend : bool, optional
            If True, plot legend, by default None.
        show : bool, optional
            If True, the plot will be shown in the console, by default True.
        kwargs : optional
            Other keywords for matplotlib. 
        Returns
        -------
        object
            axis object.
        """    
        if title is None:
            title = "Timeseries" 
        else:
            title = title
        
        if xy_labal is None:
            x_label = "Obv"; y_label = "Sim"
        else:
            x_label = xy_labal[0]; y_label = xy_labal[1]
        
        if xticks is None:
            if isinstance(x_obv, pd.DataFrame):
                xticks = x_obv.index
            else:
                xticks = np.arange(0,len(x_obv))
        else:
            assert len(xticks) == len(x_obv), print(
                "Input length of x is not corresponding to the length of data."
                )            
        fig, ax = plt.subplots()
        if isinstance(x_obv, pd.DataFrame):
            for i, c in enumerate(list(x_obv)):
                ax.plot(xticks, x_obv[c], 
                        label = x_label +"_"+ str(c), 
                        color = "C{}".format(i%10), 
                        **kwargs)
        else:
            ax.plot(xticks, x_obv, label = x_label, **kwargs)
            
        if isinstance(y_sim, pd.DataFrame):
            for i, c in enumerate(list(y_sim)):
                ax.plot(xticks, y_sim[c], linestyle='dashed', 
                        label = y_label + "_"+ str(c), 
                        color = "C{}".format(i%10), alpha = 0.5,
                        **kwargs)
        else:
            ax.plot(xticks, y_sim, linestyle='dashed', label = y_label, **kwargs)
        if legend:
            ax.legend(fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        if show:
            plt.show()
        
        if save_fig_path is not None:
            fig.savefig(save_fig_path)
            plt.close()
            
        return ax
    
    @staticmethod
    def plot_simple_ts(df, title=None, xy_labal=None, data_dots=True,
                     save_fig_path=None, **kwargs):
        """Plot timeseries.

        Parameters
        ----------
        df : DataFrame
            Dataframe.
        title : str, optional
            Title, by default None.
        xy_labal : list, optional
            List of x and y labels, by default None.
        data_dots : bool, optional
            If Ture, show data marker, by default True.
        save_fig_path : str, optional
            If given, plot will be save as .png, by default None.

        Returns
        -------
        object
            axis object.
        """
        if title is None:
            title = "" 
        else:
            title = title
        
        if xy_labal is None:
            x_label = "Time"; y_label = "Value"
        else:
            x_label = xy_labal[0]; y_label = xy_labal[1]
            
        # Create figure
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        # Regression calculation and plot
        x = np.arange(1, len(df)+1)
        for i, v in enumerate(df):
            mask = ~np.isnan(df[v])   # Mask to ignore nan
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x[mask], df[v][mask]) # Calculate the regression line
            line = slope*x+intercept  # For plotting regression line
            ax.plot(df.index, line, color="C{}".format(i%10),
                    label='y={:.2f}x+{:.2f}'.format(slope, intercept),
                    linestyle="dashed", **kwargs)
            if data_dots:
                df[[v]].plot(ax=ax, marker='o', ls='',
                             color="C{}".format(i%10), ms=2, alpha=0.6)
            else:
                df[[v]].plot(ax=ax, color="C{}".format(i%10), alpha=0.5)
        ax.legend()     
        plt.show()
        
        if save_fig_path is not None:
            fig.savefig(save_fig_path)
            plt.close()
            
        return ax
    