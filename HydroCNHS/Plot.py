
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmin
from scipy import stats
import numpy as np
import pandas as pd
from adjustText import adjust_text
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .Indicators import Indicator

class Plot():
    @staticmethod
    def RegPlot(x_obv, y_sim, Title = None, xyLabal = None, SameXYLimit = True, returnRegPar = False, SavePath = None, Show = True):
        
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
        if Show:
            plt.show()
        
        if SavePath is not None:
            fig.savefig(SavePath)
            
        if returnRegPar:
            return [slope, intercept]
        else:
            return ax

    @staticmethod
    def TimeseriesPlot(x_obv, y_sim, xticks = None, Title = None, xyLabal = None, SavePath = None, Legend = True, Show = True, **kwargs):        
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
        if Legend:
            ax.legend(fontsize=9)
        ax.set_title(Title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        #ax.set_xticks(pd.date_range(start='1/1/1966', end='12/31/2005'))
        if Show:
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
    def EquifinalPlot(Caliobj, k, SelectedPar = None, q = 0.01, SavePath = None, Rotation = 90, AdjustText = True, Sep = False):
        
        if isinstance(Caliobj, list):
            Name = "Collective Caliobj"
            df = pd.DataFrame()
            for caliobj in Caliobj:
                KPopResult = caliobj.KPopRes
                MaxGen = caliobj.Config["MaxGen"]
                PopSize = caliobj.Config["PopSize"]
                NumPar = caliobj.NumPar
                Pop = caliobj.Pop
                ParName = caliobj.Inputs["ParName"]
                Bound = caliobj.Inputs["ParBound"]
                ParWeight = caliobj.Inputs["ParWeight"]
                if SelectedPar is None:
                    SelectedPar = ParName
                
                # Get feasible loss
                Loss = np.zeros((MaxGen+1)*PopSize)
                PopAll = np.zeros(((MaxGen+1)*PopSize, NumPar))
                for i in range(MaxGen+1):
                    Loss[i*PopSize:(i+1)*PopSize] = KPopResult[i]["Loss"]
                    PopAll[i*PopSize:(i+1)*PopSize,:] = Pop[i]
                dff = pd.DataFrame(PopAll, columns = ParName)
                dff["Loss"] = Loss
                df = pd.concat([df,dff], axis = 0)
        else:
            KPopResult = Caliobj.KPopRes
            MaxGen = Caliobj.Config["MaxGen"]
            PopSize = Caliobj.Config["PopSize"]
            NumPar = Caliobj.NumPar
            Pop = Caliobj.Pop
            ParName = Caliobj.Inputs["ParName"]
            Bound = Caliobj.Inputs["ParBound"]
            ParWeight = Caliobj.Inputs["ParWeight"]
            Name = Caliobj.__name__
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
        
        # Sort by Loss, so we can keep the better Loss of same par set in next step.
        df = df.sort_values(by='Loss', ascending=True).reset_index(drop=True)     
        # Remove the duplicates (keep the first one (smallest) if same par but different loss)
        df.drop_duplicates(subset=list(df)[:-1], keep='first', inplace=True, ignore_index=True)
        Loss_q = np.quantile(df["Loss"], q)
        
        # Get feasible pop
        df = df[df["Loss"] <= Loss_q]
        df = df[SelectedPar + ["Loss"]]
        
        # Normalize Loss for plotting.
        MaxLoss = round(max(df["Loss"])*1.2, 1)
        MinLoss = round(min(df["Loss"])*0.8, 1)
        df["Loss"] = (df["Loss"] - MinLoss)/(MaxLoss - MinLoss)
        
        # Get best
        Bestpop = df[df["Loss"] == min(df["Loss"])][SelectedPar + ["Loss"]]
        if Bestpop.shape[0] > 1:
            print("We have {} Bestpop with same loss value. We will pick the first one to plot.".format(Bestpop.shape[0]))
            Bestpop = Bestpop.iloc[0,:]
            
        # Run Kmeans
        km = KMeans(n_clusters = k, random_state=0).fit(df[SelectedPar], ParWeight)
        df["Label"] = km.labels_
        Centers = km.cluster_centers_
        
        # Plot
        # if FigSize is not None:
        #     fig, ax = plt.subplots(figsize = FigSize)
        # else:
        #     fig, ax = plt.subplots()
        if Sep is False:
            Height = 8
            FigSize=((len(SelectedPar)+1)/16*Height, Height)
            fig, ax = plt.subplots(figsize = FigSize)
            
            dfBound = pd.DataFrame(Bound, index = ParName, columns = ["LB", "UB"])
            dfBound = dfBound.loc[SelectedPar,:]
            
            
            for i in range(k):
                df_k = df[df["Label"] == i][SelectedPar + ["Loss"]].T
                df_k.plot(lw = 0.5, color = "C{}".format(i%10), alpha = 0.2, legend = False, ax=ax)
                
                dff = df[df["Label"] == i]
                minLoss = min(dff["Loss"])
                center = Centers[i]
                dff = dff[dff["Loss"] == minLoss]
                dff = dff.reset_index(drop=True)
                # Calculate distance to the center
                selectpar = dff[SelectedPar].to_numpy()
                Dist = []
                for i in selectpar:
                    Dist.append(np.linalg.norm(i-center))
                index = argmin(Dist)
                ax.plot(dff.loc[index, SelectedPar + ["Loss"]].to_numpy().T,lw = 1, color = "black", linestyle = "-")
    
                
                
            ax.plot(Bestpop.to_numpy().T,lw = 1.2, color = "red", linestyle = "--")
            
            
            ax.set_xticks(np.arange(len(SelectedPar) + 1)) 
            ax.set_xticklabels(SelectedPar + ["Loss"], fontsize=12)
            ax.set_yticks([])
            ax.tick_params(axis='x', rotation=Rotation, labelsize = 12)
            ax.axhline(0, color = "black", lw = 0.5)
            ax.axhline(1, color = "black", lw = 0.5)
            ax.set_title(Name + "    Thres: {}".format(round(Loss_q,3)))    # Use the original Loss from Caliobj
            ax.set_ylim([-0.1,1.1])
            def sn(num):
                '''Control length of printing number'''
                Num = ('%f' % num).rstrip('0').rstrip('.')
                if len(Num) > 4:
                    Num = str(np.format_float_scientific(num, exp_digits=1, trim = "-"))
                return Num
            Texts = []
            for x in range(len(SelectedPar)):
                Texts.append(ax.text(x, -0.05, sn(dfBound["LB"][x]), horizontalalignment='center', fontsize=10))
                Texts.append(ax.text(x, 1.05, sn(dfBound["UB"][x]), horizontalalignment='center', fontsize=10))
                ax.axvline(x, color = "grey", lw = 0.2)
            # Add Loss
            x = len(SelectedPar)
            Texts.append(ax.text(x, -0.05, sn(MinLoss), horizontalalignment='center', fontsize=10))
            Texts.append(ax.text(x, 1.05, sn(MaxLoss), horizontalalignment='center', fontsize=10))
            ax.axvline(x, color = "grey", lw = 0.2)
                
            # Auto adjust label position.
            if AdjustText:
                print("Auto text position adjustment might take some time.")
                adjust_text(Texts, only_move={"points":"y", "text":"y", "objects":"y"}, expand_text =(1,1))
                
            if SavePath is not None:
                fig.savefig(SavePath)
            return ax
        else:
            for i in range(k):
                Height = 8
                FigSize=((len(SelectedPar)+1)/16*Height, Height)
                fig, ax = plt.subplots(figsize = FigSize)
                
                dfBound = pd.DataFrame(Bound, index = ParName, columns = ["LB", "UB"])
                dfBound = dfBound.loc[SelectedPar,:]
            
                df_k = df[df["Label"] == i][SelectedPar + ["Loss"]].T
                df_k.plot(lw = 0.5, color = "C{}".format(i%10), alpha = 0.2, legend = False, ax=ax)
                
                dff = df[df["Label"] == i]
                minLoss = min(dff["Loss"])
                center = Centers[i]
                dff = dff[dff["Loss"] == minLoss]
                dff = dff.reset_index(drop=True)
                # Calculate distance to the center
                selectpar = dff[SelectedPar].to_numpy()
                Dist = []
                for i in selectpar:
                    Dist.append(np.linalg.norm(i-center))
                index = argmin(Dist)
                ax.plot(dff.loc[index, SelectedPar + ["Loss"]].to_numpy().T,lw = 1, color = "black", linestyle = "-")
    
                #ax.plot(Bestpop.to_numpy().T,lw = 1.2, color = "red", linestyle = "--")
                
                ax.set_xticks(np.arange(len(SelectedPar) + 1)) 
                ax.set_xticklabels(SelectedPar + ["Loss"], fontsize=12)
                ax.set_yticks([])
                ax.tick_params(axis='x', rotation=Rotation, labelsize = 12)
                ax.axhline(0, color = "black", lw = 0.5)
                ax.axhline(1, color = "black", lw = 0.5)
                ax.set_title(Name + "  Group: {}  Thres: {}".format(k+1, round(Loss_q,3)))    # Use the original Loss from Caliobj
                ax.set_ylim([-0.1,1.1])
                def sn(num):
                    '''Control length of printing number'''
                    Num = ('%f' % num).rstrip('0').rstrip('.')
                    if len(Num) > 4:
                        Num = str(np.format_float_scientific(num, exp_digits=1, trim = "-"))
                    return Num
                Texts = []
                for x in range(len(SelectedPar)):
                    Texts.append(ax.text(x, -0.05, sn(dfBound["LB"][x]), horizontalalignment='center', fontsize=10))
                    Texts.append(ax.text(x, 1.05, sn(dfBound["UB"][x]), horizontalalignment='center', fontsize=10))
                    ax.axvline(x, color = "grey", lw = 0.2)
                # Add Loss
                x = len(SelectedPar)
                Texts.append(ax.text(x, -0.05, sn(MinLoss), horizontalalignment='center', fontsize=10))
                Texts.append(ax.text(x, 1.05, sn(MaxLoss), horizontalalignment='center', fontsize=10))
                ax.axvline(x, color = "grey", lw = 0.2)
                    
                # Auto adjust label position.
                if AdjustText:
                    print("Auto text position adjustment might take some time.")
                    adjust_text(Texts, only_move={"points":"y", "text":"y", "objects":"y"}, expand_text =(1,1))
                    
        
    
    @staticmethod
    def YearPlot(df, ylim = None, **kwargs):
        axs = []
        for i in df:
            fig, ax = plt.subplots()
            x = np.arange(1, 13)
            S = df.index[0].year
            E = df.index[-1].year
            for y in range(S, E+1):
                dff = df[df.index.year == y]
                ax.plot(x, dff[i], color = "grey", lw = 1, alpha = 0.3, **kwargs)
                ax.set_title(i)
            if ylim is not None:
                ax.set_ylim(ylim)
            axs.append(ax)
        return axs
                
    @staticmethod
    def getEquifinalModels(Caliobj, KClusterMin = 1, KClusterMax = 10, k = None, SelectedPar = None, q = 0.01, TakeBest = True, bins = 50):
        
        if isinstance(Caliobj, list):
            df = pd.DataFrame()
            for caliobj in Caliobj:
                KPopResult = caliobj.KPopRes
                MaxGen = caliobj.Config["MaxGen"]
                PopSize = caliobj.Config["PopSize"]
                NumPar = caliobj.NumPar
                Pop = caliobj.Pop
                ParName = caliobj.Inputs["ParName"]
                Bound = caliobj.Inputs["ParBound"]
                ParWeight = caliobj.Inputs["ParWeight"]
                if SelectedPar is None:
                    SelectedPar = ParName
                
                # Get feasible loss
                Loss = np.zeros((MaxGen+1)*PopSize)
                PopAll = np.zeros(((MaxGen+1)*PopSize, NumPar))
                for i in range(MaxGen+1):
                    Loss[i*PopSize:(i+1)*PopSize] = KPopResult[i]["Loss"]
                    PopAll[i*PopSize:(i+1)*PopSize,:] = Pop[i]
                dff = pd.DataFrame(PopAll, columns = ParName)
                dff["Loss"] = Loss
                df = pd.concat([df,dff], axis = 0)
        else:
            caliobj = Caliobj
            KPopResult = Caliobj.KPopRes
            MaxGen = Caliobj.Config["MaxGen"]
            PopSize = Caliobj.Config["PopSize"]
            NumPar = Caliobj.NumPar
            Pop = Caliobj.Pop
            ParName = Caliobj.Inputs["ParName"]
            Bound = Caliobj.Inputs["ParBound"]
            ParWeight = Caliobj.Inputs["ParWeight"]
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
            
        # Process
        # Sort by Loss, so we can keep the better Loss of same par set in next step.
        df = df.sort_values(by='Loss', ascending=True).reset_index(drop=True)     
        # Remove the duplicates (keep the first one (smallest) if same par but different loss)
        df.drop_duplicates(subset=list(df)[:-1], keep='first', inplace=True, ignore_index=True)
        Loss_q = np.quantile(df["Loss"], q)

        # Get feasible pop
        # plt.hist(df["Loss"], bins = bins)
        # plt.title("Histogram with all simulations ({})".format(len(df["Loss"])))
        # plt.show()
        
        df = df[df["Loss"] <= Loss_q]
        df = df[SelectedPar + ["Loss"]]
        plt.hist(df["Loss"], bins = bins)
        plt.title("Histogram ({})".format(len(df["Loss"])))
        plt.show()
        
        if k is None:
            KClusterMax = 10
            KClusterMin = 1
            KmeansModel = {}
            KDistortions = []
            KExplainedVar = []
            SilhouetteAvg = []
            SSE = np.sum(np.var(df[SelectedPar], axis = 0))*df[SelectedPar].shape[0]
            for k in range(KClusterMin, KClusterMax+1):
                km = KMeans(n_clusters = k, random_state=0).fit(df[SelectedPar], ParWeight)
                KmeansModel[k] = km
                # Calculate some indicators for kmeans
                ## inertia_: Sum of squared distances of samples to their closest cluster center.
                KDistortions.append(km.inertia_)
                KExplainedVar.append((SSE - KDistortions[-1])/SSE)
                ## The silhouette_score gives the average value for all the samples.
                ## This gives a perspective into the density and separation of the formed clusters
                ## The coefficient varies between -1 and 1. A value close to 1 implies that the instance is close to its cluster is a part of the right cluster. 
                cluster_labels = km.labels_
                if k == 1:  # If given k == 1, then assign the worst value.
                    SilhouetteAvg.append(-1) 
                else:
                    silhouette_avg = silhouette_score(df[SelectedPar], cluster_labels)
                    SilhouetteAvg.append(silhouette_avg)
            fig, ax = plt.subplots()
            x = np.arange(KClusterMin, KClusterMax+1)
            ax.plot(x, SilhouetteAvg, "+-", label = "Silhouette Score")
            ax.plot(x, KExplainedVar, "+-", label = "Explained Var")
            ax.legend() 
            plt.show()
        else:
            # Run Kmeans
            km = KMeans(n_clusters = k, random_state=0).fit(df[SelectedPar], ParWeight)
            df["Label"] = km.labels_
            Centers = km.cluster_centers_
            
            # Take out the best-performed model in each kmeans group.
            # If multiple equal loss model, then select the most closed to the center one.
            EquifinalDF = pd.DataFrame(index=ParName)
            for g in range(k):
                center = Centers[g]
                dff = df[df["Label"] == g]
                if TakeBest:
                    minLoss = min(dff["Loss"])
                    dff = dff[dff["Loss"] == minLoss]
                dff = dff.reset_index(drop=True)
                # Calculate distance to the center
                selectpar = dff[SelectedPar].to_numpy()
                Dist = []
                for i in selectpar:
                    Dist.append(np.linalg.norm(selectpar-center))
                index = argmin(Dist)
                EquifinalDF[g] = caliobj.scale(dff.loc[index, ParName].to_numpy().flatten())
            return EquifinalDF
