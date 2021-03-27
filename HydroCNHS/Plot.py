
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from .Indicators import Indicator

class Plot():
    @staticmethod
    def RegPlot(x_obv, y_sim, Title = None, xyLabal = None, SameXYLimit = True):
        
        if Title is None:
            Title = "Regression" 
        else:
            Title = Title
        
        if xyLabal is None:
            x_label = "Obv"; y_label = "Sim"
        else:
            x_label = xyLabal[0]; y_label = xyLabal[1]
            
        # Create figure
        fig, ax = plt.subplots()
        ax.set_title(Title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        # Regression calculation and plot
        mask = ~np.isnan(x_obv) & ~np.isnan(y_sim)  # Mask to ignore nan
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_obv[mask], y_sim[mask]) # Calculate the regression line
        line = slope*x_obv+intercept                # For plotting regression line
        ax.plot(x_obv, line, 'r', label='y={:.2f}x+{:.2f}'.format(slope, intercept))
        # end
        
        # # Indicator calculation
        r = np.corrcoef(x_obv[mask], y_sim[mask])[0,1]
        r2 = r**2
        rmse = np.nanmean((x_obv[mask]-y_sim[mask])**2)**0.5
        CE = 1 - np.nansum((x_obv[mask]-y_sim[mask])**2)/np.nansum((x_obv[mask]-np.nanmean(x_obv[mask]))**2) # Nash
        CP = 1 - np.nansum((x_obv[1:]-y_sim[1:])**2)/np.nansum((x_obv[1:]-x_obv[:-1])**2)
        
        # Plot data point
        ax.scatter(x_obv, y_sim, color="k", s=3.5)
        ax.legend(fontsize=9, loc = 'upper right')
        if SameXYLimit:
            Max = max([np.nanmax(x_obv),np.nanmax(y_sim)]); Min = min([np.nanmin(x_obv),np.nanmin(y_sim)])
            ax.set_xlim(Min,Max)
            ax.set_ylim(Min,Max)
            
        # PLot indicators
        Name = {"r": "$r$",
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
        
        string = "\n".join(['{:^4}: {}'.format(Name[keys], round(values,5)) for keys,values in indicators.items()])
        
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
        ax.annotate(string, xy= (0.05, 0.95), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, fontsize=9, bbox = props)       
        plt.show()
        return ax

    @staticmethod
    def TimeseriesPlot(x_obv, y_sim, xticks = None, Title = None, xyLabal = None, **kwargs):        
        if Title is None:
            Title = "Timeseries" 
        else:
            Title = Title
        
        if xyLabal is None:
            x_label = "Obv"; y_label = "Sim"
        else:
            x_label = xyLabal[0]; y_label = xyLabal[1]
        
        if xticks is None:
            xticks = np.arange(0,len(x_obv))
        else:
            assert len(xticks) == len(x_obv), print("Input length of x is not corresponding to the length of data.")
            # try:
            #     x = pd.date_range(start=x[0], end=x[1])
            # except:
            #     x = x
                
        fig, ax = plt.subplots()
        if isinstance(x_obv, pd.DataFrame):
            for c in list(x_obv):
                ax.plot(xticks, x_obv[c], label = x_label +"_"+ str(c), **kwargs)
        else:
            ax.plot(xticks, x_obv, label = x_label, **kwargs)
            
        if isinstance(y_sim, pd.DataFrame):
            for c in list(y_sim):
                ax.plot(xticks, y_sim[c], linestyle='dashed', label = y_label + "_"+ str(c), **kwargs)
        else:
            ax.plot(xticks, y_sim, linestyle='dashed', label = y_label, **kwargs)
        #ax.bar(x, np.nan_to_num(y_obv-y_sim), label = "Hydromet - YAKRW", color = "red")
        ax.legend(fontsize=9)
        ax.set_title(Title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        #ax.set_xticks(pd.date_range(start='1/1/1966', end='12/31/2005'))
        plt.show()
        return ax