
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .Indicators import Indicator

class Plot():
    @staticmethod
    def RegPlot(x_obv, y_sim, Title = None, xyLabal = None, SameXYLimit = True, returnRegPar = False, SavePath = None):
        
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
        x_obv = np.array(x_obv)
        y_sim = np.array(y_sim)
        mask = ~np.isnan(x_obv) & ~np.isnan(y_sim)  # Mask to ignore nan
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_obv[mask], y_sim[mask]) # Calculate the regression line
        line = slope*x_obv+intercept                # For plotting regression line
        ax.plot(x_obv, line, 'r', label='y={:.2f}x+{:.2f}'.format(slope, intercept))
        # end
        
        # Plot data point
        ax.scatter(x_obv, y_sim, color="k", s=3.5)
        ax.legend(fontsize=9, loc = 'upper right')
        if SameXYLimit:
            Max = max([np.nanmax(x_obv),np.nanmax(y_sim)]); Min = min([np.nanmin(x_obv),np.nanmin(y_sim)])
            ax.set_xlim(Min,Max)
            ax.set_ylim(Min,Max)
            # Add 45 degree line
            interval = (Max-Min)/10
            Diagonal = np.arange(Min,Max+interval,interval)
            ax.plot(Diagonal, Diagonal, "b", linestyle='dashed', lw = 1)
        
        
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
        
        if SavePath is not None:
            fig.savefig(SavePath)
            
        if returnRegPar:
            return [slope, intercept]
        else:
            return ax

    @staticmethod
    def TimeseriesPlot(x_obv, y_sim, xticks = None, Title = None, xyLabal = None, SavePath = None, **kwargs):        
        if Title is None:
            Title = "Timeseries" 
        else:
            Title = Title
        
        if xyLabal is None:
            x_label = "Obv"; y_label = "Sim"
        else:
            x_label = xyLabal[0]; y_label = xyLabal[1]
        
        if xticks is None:
            if isinstance(x_obv, pd.DataFrame):
                xticks = x_obv.index
            else:
                xticks = np.arange(0,len(x_obv))
        else:
            assert len(xticks) == len(x_obv), print("Input length of x is not corresponding to the length of data.")
            # try:
            #     x = pd.date_range(start=x[0], end=x[1])
            # except:
            #     x = x
                
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
        #ax.bar(x, np.nan_to_num(y_obv-y_sim), label = "Hydromet - YAKRW", color = "red")
        ax.legend(fontsize=9)
        ax.set_title(Title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        #ax.set_xticks(pd.date_range(start='1/1/1966', end='12/31/2005'))
        plt.show()
        
        if SavePath is not None:
            fig.savefig(SavePath)
            
        return ax
    
    @staticmethod
    def SimpleTSPlot(df, Title = None, xyLabal = None, Dot = True, SavePath = None, **kwargs):
        if Title is None:
            Title = "" 
        else:
            Title = Title
        
        if xyLabal is None:
            x_label = "Time"; y_label = "Value"
        else:
            x_label = xyLabal[0]; y_label = xyLabal[1]
            
        # Create figure
        fig, ax = plt.subplots()
        ax.set_title(Title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        # Regression calculation and plot
        x = np.arange(1, len(df)+1)
        for i, v in enumerate(df):
            mask = ~np.isnan(df[v])  # Mask to ignore nan
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], df[v][mask]) # Calculate the regression line
            line = slope*x+intercept                # For plotting regression line
            ax.plot(df.index, line, color = "C{}".format(i%10), label='y={:.2f}x+{:.2f}'.format(slope, intercept), linestyle = "dashed", **kwargs)
            if Dot:
                df[[v]].plot(ax=ax, marker='o', ls='', color = "C{}".format(i%10), ms=2, alpha = 0.6)
            else:
                df[[v]].plot(ax=ax, color = "C{}".format(i%10), alpha = 0.5)
        ax.legend()     
        plt.show()
        
        if SavePath is not None:
            fig.savefig(SavePath)
            
        return ax
    
    @staticmethod
    def EquifinalPlot(Caliobj, k, SelectedPar = None, q = 0.01, SavePath = None):
        KPopResult = Caliobj.KPopRes
        MaxGen = Caliobj.Config["MaxGen"]
        PopSize = Caliobj.Config["PopSize"]
        NumPar = Caliobj.NumPar
        Pop = Caliobj.Pop
        ParName = Caliobj.Inputs["ParName"]
        Bound = Caliobj.Inputs["ParBound"]
        if SelectedPar is None:
            SelectedPar = ParName
        
        # Get feasible loss
        Loss = np.zeros((MaxGen+1)*PopSize)
        PopAll = np.zeros(((MaxGen+1)*PopSize, NumPar))
        for i in range(MaxGen+1):
            Loss[i*PopSize:(i+1)*PopSize] = KPopResult[i]["Loss"]
            PopAll[i*PopSize:(i+1)*PopSize,:] = Pop[i]
        df = pd.DataFrame(PopAll, columns = ParName)
        df["Loss"] = Loss
        df = df.drop_duplicates().reset_index(drop=True)   # Remove the duplicates
        Loss_q = np.quantile(df["Loss"], q)

        
        # Get feasible pop
        df = df[df["Loss"] <= Loss_q]
        df = df[SelectedPar + ["Loss"]]

        # Get best
        Bestpop = Caliobj.descale(Caliobj.Result["GlobalOptimum"]["Solutions"])
        
        # Run Kmeans
        ParWeight = Caliobj.Inputs["ParWeight"]
        km = KMeans(n_clusters = k, random_state=0).fit(df[SelectedPar], ParWeight)
        df["Label"] = km.labels_
        
        # Plot
        fig, ax = plt.subplots()
        
        dfBound = pd.DataFrame(Bound, index = ParName, columns = ["LB", "UB"])
        dfBound = dfBound.loc[SelectedPar,:]
        
        
        for i in range(k):
            df_k = df[df["Label"] == i][SelectedPar].T
            #df_k = df_k[SelectedPar + ["Loss"]]
            df_k.plot(lw = 0.3, color = "C{}".format(i%10), alpha = 0.3, legend = False, ax=ax)
            
        ax.plot(Bestpop,lw = 0.5, color = "red", linestyle = "--")
        ax.set_xticks(np.arange(len(SelectedPar))) 
        ax.set_xticklabels(SelectedPar, fontsize=10)
        ax.set_yticks([])
        ax.tick_params(axis='x', rotation=30, labelsize = 8)
        ax.axhline(0, color = "black", lw = 0.5)
        ax.axhline(1, color = "black", lw = 0.5)
        ax.set_title(Caliobj.__name__ + "    Thres: {}".format(round(1-Loss_q,3)))
        ax.set_ylim([-0.1,1.1])
        for x in range(len(SelectedPar)):
            ax.text(x, -0.05, dfBound["LB"][x], horizontalalignment='center', fontsize=6)
            ax.text(x, 1.05, dfBound["UB"][x], horizontalalignment='center', fontsize=6)
            ax.axvline(x, color = "grey", lw = 0.1)
            
        if SavePath is not None:
            fig.savefig(SavePath)
        
        return ax