#%%
# Kmeans Genetic Calibration Algorithm (KGCA).
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# The algorithm is based on the idea of (Poikolainen et al., 2015) DOI: 10.1016/j.ins.2014.11.026.
# However, we simplify and modify the idea to fit our need for identifying equifinal model representatives (EMRs). 
# 2021/03/13

from scipy.stats import rankdata, truncnorm             # Rank data & Truncated normal distribution.
from joblib import Parallel, delayed                    # For parallelization.
from sklearn.cluster import KMeans                      # KMeans algorithm by sklearn.
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import logging
import pickle
import time
import os
logger = logging.getLogger("HydroCNHS.KGCA")            # Get logger 

from ..SystemConrol import loadConfig, Dict2String      # HydroCNHS module

r"""
Inputs = {"ParName":    [],     # List of parameters name.
          "ParBound":   [],     # [upper, low] or [4, 6, 9] Even for categorical type, it has to be numbers!
          "ParType":    [],     # real or categorical
          #"ParWeight": [],     # An array with length equal to number of parameters.
          "WD":         r""}    # Working directory. (Can be same as HydroCNHS model.)   
# Note: A more convenient way to generate the Inputs dictionary is to use "Convertor" provided by HydroCNHS.Cali.

Config = {"PopSize":            30,     # Population size. Must be even.
          "ParantProportion":   0.3,
          "NumEllites":         1,
          "CrossProb":          0.5,
          "Stochastic":         False,
          "MaxGen":             100,    # Maximum generation.
          "SamplingMethod":     "LHC",  # MC: Monte Carlo sampling method. LHC: Latin Hyper Cube. (for initial pop)
          "MutProb":            0.3,    # Mutation probability.
          "DropRecord":         True,   # Population record will be dropped. However, ALL simulated results will still be kept. 
          "ParalCores":         2/None, # This will replace system config.
          "AutoSave":           True,   # Automatically save a model snapshot after each generation.
          "Printlevel":         10,     # Print out level. e.g. Every ten generations.
          "Plot":               True    # Plot loss and cluster number selection with Printlevel frequency.
          }
"""

class KGCASimple(object):
    def __init__(self, LossFunc, Inputs, Config, Formatter = None, ContinueFile = None, Name = "Calibration"):
        """Kmeans Genetic Calibration Algorithm (KGCA)

        Args:
            LossFunc (function): Loss function => LossFunc(pop, Formatter, SubWDInfo = None) and return loss, which has lower bound 0. SubWDInfo = (CaliWD, CurrentGen, sp, k).
            Inputs (dict): Inputs dictionary, which can be generated by Convertor. It contains ParName, ParBound, ParType, ParWeight, and WD.
            Config (dict): Config dictionary
            Formatter (dict, optional): Formatter dictionary created by Convertor. This will be fed to LossFunc for users to convert 1D array (pop) back to original format to run HydroCNHS. Defaults to None.
            ContinueFile (str, optional): AutoSave.pickle directory to continue the previous run. Defaults to None.
            Name (str, optional): Name of the KGCA object. The sub-folder will be created accordingly. Defaults to "Calibration".
        """
               
        # Populate class attributions.
        self.LossFunc = LossFunc                # Loss function LossFunc(pop, Formatter, SubWDInfo = None) return loss, which has lower bound 0.
                                                    # pop is a parameter vector. 
                                                    # Formatter can be obtained from class Convertor.
                                                    # SubWDInfo = (CaliWD, CurrentGen, sp, k).
                                                    # Lower bound of return value, which has to be 0.
        self.Inputs = Inputs                    # Input ductionary.
        self.Inputs["ParWeight"] = np.array(Inputs["ParWeight"])    # Make sure it is an array.
        self.Config = Config                    # Configuration for KGCA.
        self.Formatter = Formatter              # Formatter is to convert 1D pop back into list of dataframe dictionaries for HydroCNHS simulation. (see class Convertor) 
        self.SysConfig = loadConfig()           # Load system config => Config.yaml (Default parallelization setting)
        self.NumPar = len(Inputs["ParName"])    # Number of calibrated parameters.
        self.ParantSize = self.Config["ParantProportion"]*self.Config["PopSize"]
        trl = self.par['PopSize'] - self.ParantSize
        if trl % 2 != 0: 
            self.ParantSize -= 1  # To guarentee even number 
    
        # If ContinueFile is given, load auto-saved pickle file.
        if ContinueFile is not None:
            with open(ContinueFile, "rb") as f:     # Load autoSave pickle file!
                Snapshot = pickle.load(f)
            for key in Snapshot:                    # Load back all the previous class attributions.
                setattr(self, key, Snapshot[key])
            self.Continue = True                    # No initialization is needed in "run", when self.Continue = True.

        #---------- Auto save section ----------
        if ContinueFile is None:
            self.Continue = False                   # Initialization is required in "run", when self.Continue = False.            
            self.CurrentGen = 0                     # Populate initial counter for generation.
            
            # Initialize variables storage.
            self.Pop = {}           # Population of parameter set. Pop[gen][sp]: [k,s] (2D array); k is index of members, and s is index of parameters.
            self.PopRes = {}        # Simulation results of each members. PopRes[gen][sp][Loss/Dmin/Feasibility/SelfD]: 1D array with length of population size.
            self.KPopRes = {}       # Store Kmeans results
            
            # Best loss value and index of corresponding member in Pop[gen][sp]. Best[Loss/Index][sp]: 1D array with length of MaxGen.
            self.Best = {"Loss":  np.empty(self.Config["MaxGen"]+1),    # +1 since including gen 0.
                         "Index": np.empty(self.Config["MaxGen"]+1)}
            
            # Calculate scales for parameter normalization.
            # We assume categorical type is still number kind list (e.g. [1,2,3,4] and scale = 4-1 = 3).  
            self.BoundScale = []
            self.LowerBound = []
            for i, ty in enumerate(Inputs["ParType"]):
                if ty == "real":
                    self.BoundScale.append(Inputs["ParBound"][i][1] - Inputs["ParBound"][i][0])
                    self.LowerBound.append(Inputs["ParBound"][i][0])
                # elif ty == "categorical":
                #     self.BoundScale.append(np.max(Inputs["ParBound"][i]) - np.min(Inputs["ParBound"][i]))
                #     self.LowerBound.append(Inputs["ParBound"][i][0])
            self.BoundScale = np.array(self.BoundScale).reshape((-1,self.NumPar))     # Store in an array type. 
            self.LowerBound = np.array(self.LowerBound).reshape((-1,self.NumPar))
            
            # Create calibration folder under WD
            self.__name__ = Name
            self.CaliWD = os.path.join(Inputs["WD"], self.__name__)
            # Create CaliWD directory
            if os.path.isdir(self.CaliWD) is not True:
                os.mkdir(self.CaliWD)
            else:
                logger.warning("\n[!!!Important!!!] Current calibration folder exists. Default to overwrite the folder!\n{}".format(self.CaliWD))
        #---------------------------------------
        
    def scale(self, pop):
        """pop is 1d array (self.NumPar) or 2d array (-1,self.NumPar)."""
        pop = pop.reshape((-1,self.NumPar))
        BoundScale = self.BoundScale    # (-1,self.NumPar)
        LowerBound = self.LowerBound    # (-1,self.NumPar)
        ScaledPop = np.multiply(pop, BoundScale)
        ScaledPop = np.add(ScaledPop, LowerBound)
        if ScaledPop.shape[0] == 1:
            ScaledPop = ScaledPop[0]    # Back to 1D.
        return ScaledPop
    
    def descale(self, pop):
        """pop is 1d array (self.NumPar) or 2d array (-1,self.NumPar)."""
        pop = pop.reshape((-1,self.NumPar))
        BoundScale = self.BoundScale    # (-1,self.NumPar)
        LowerBound = self.LowerBound    # (-1,self.NumPar)
        deScaledPop = np.subtract(pop, LowerBound)
        deScaledPop = np.divide(deScaledPop, BoundScale)
        
        if deScaledPop.shape[0] == 1:
            deScaledPop = deScaledPop[0]    # Back to 1D.
        return deScaledPop
    
    def MCSample(self, pop):
        """Generate samples using Monte Carlo method. Only real par type.

        Args:
            pop (Array): 2D array. [PopSize, NumPar]
            
        Returns:
            array: Populated pop array.
        """
        PopSize = pop.shape[0]      # pop = [PopSize, NumPar]
        NumPar = pop.shape[1]
        for i in range(NumPar):
            pop[:,i] = np.random.uniform(0, 1, size = PopSize)  
        return pop 
    
    def LatinHyperCubeSample(self, pop):
        """Generate samples using Latin Hyper Cube (LHC) method. Only real par type.

        Args:
            pop (Array): 2D array. [PopSize, NumPar]

        Returns:
            array: Populated pop array.
        """
        PopSize = pop.shape[0]      # pop = [PopSize, NumPar]
        NumPar = pop.shape[1]
        for i in range(NumPar):
            d = 1.0 / PopSize
            temp = np.empty([PopSize])
            # Uniformly sample in each interval.
            for j in range(PopSize):
                temp[j] = np.random.uniform(low=j * d, high=(j + 1) * d)
            # Shuffle to random order.
            np.random.shuffle(temp)
            # Scale [0,1] to its bound.
            pop[:,i] = temp
        return pop 
      
    def initialize(self, SamplingMethod = "LHC", InitialPop = None):
        """Initialize population members and storage spaces (KPopRes) for generation 0.

        Args:
            SamplingMethod (str, optional): Selected method for generate initial population members. MC or LHC.
            InitialPop (dict, optional): User-provided initial Pop[0] = [-1, NumPar] (2D array). Note that InitialPop will be descaled to [0,1] Defaults to None.
        """
        PopSize = self.Config["PopSize"]
        NumPar = self.NumPar

        # Note Pop is a scaled values in [0, 1]
        if InitialPop is None:      
            # Initialize storage space for generation 0 (2D array).
            pop = np.zeros((PopSize, NumPar))  
            ## Initialize parameters according to selected sampling method.
            if SamplingMethod == "MC":
                self.Pop[0] = self.MCSample(pop)
            elif SamplingMethod == "LHC":
                self.Pop[0] = self.LatinHyperCubeSample(pop)
        else:                       
            if PopSize == InitialPop.shape[0] and NumPar == InitialPop.shape[1]:
                self.Pop[0] = self.descale(InitialPop)  # to [0,1]
            else:   
                # Initialize storage space for generation 0 (2D array) and assign InitialPop.
                InitialPop = self.descale(InitialPop)   # to [0,1]
                self.Pop[0] = np.zeros((PopSize, NumPar)) 
                self.Pop[0][0:InitialPop[0], :] = InitialPop
                # Sample the rest of the pop
                pop_s = np.zeros((PopSize-InitialPop.shape[0], NumPar))  
                ## Initialize parameters according to selected sampling method.
                if SamplingMethod == "MC":
                    self.Pop[0][InitialPop[0]:, :] = self.MCSample(pop_s)
                elif SamplingMethod == "LHC":
                    self.Pop[0][InitialPop[0]:, :] = self.LatinHyperCubeSample(pop_s)

        # Initialize storage space.
        self.KPopRes[0] = {}
        
    def nextGen(self):
        """Complete all simulations and generate next generation of Pop.
        """
        LossFunc = self.LossFunc
        Formatter = self.Formatter
        CurrentGen = self.CurrentGen
        PopSize = self.Config["PopSize"]
        NumPar = self.NumPar

        # Load parallelization setting (from user or system config)
        ParalCores = self.Config.get("ParalCores")
        if ParalCores is None:      # If user didn't specify ParalCores, then we will use default cores in the system config.
            ParalCores = self.SysConfig["Parallelization"]["Cores_KGCA"]
        ParalVerbose = self.SysConfig["Parallelization"]["verbose"]         # Joblib print out setting.
        
        #---------- Evaluation (Min) ----------
        # Note: Since HydroCHNS is a stochastic model, we will re-simulate ellites!!
        # In future, we can have an option for this (re-simulate ellites) to further enhance computational efficiency.
        # Evalute Loss function
        # LossFunc(pop, Formatter, SubWDInfo = None) and return loss, which has lower bound 0.
        ScaledPop = self.scale(self.Pop[CurrentGen])    # Scale back to original values.
        if self.Config["Stochastic"] or CurrentGen == 0:
            LossParel = Parallel(n_jobs = ParalCores, verbose = ParalVerbose) \
                            ( delayed(LossFunc)\
                                (ScaledPop[k], Formatter, (self.CaliWD, CurrentGen, "", k)) \
                                for k in range(PopSize) )  # Still go through entire Pop including ellites.
            # Get loss results
            Loss = np.array(LossParel[0:PopSize])
            self.KPopRes[CurrentGen]["Loss"] = Loss
        else:
            NumEffParents = self.NumEffParents
            LossParel = Parallel(n_jobs = ParalCores, verbose = ParalVerbose) \
                            ( delayed(LossFunc)\
                                (ScaledPop[k], Formatter, (self.CaliWD, CurrentGen, "", k)) \
                                for k in range(NumEffParents, PopSize) )  # Still go through entire Pop including ellites.
            # Get loss results
            Loss = np.empty(PopSize)
            Loss[0:NumEffParents] = self.EffLoss    # Add previous losses.
            Loss[NumEffParents:PopSize] = np.array(LossParel[0:PopSize-NumEffParents])
            self.KPopRes[CurrentGen]["Loss"] = Loss
            
        # Record the current global optimum. 
        BestIndex = np.argmin(self.KPopRes[CurrentGen]["Loss"])
        Best = self.KPopRes[CurrentGen]["Loss"][BestIndex]
        self.Best["Loss"][CurrentGen] = Best            # Array
        self.Best["Index"][CurrentGen] = BestIndex      # Array
        
        # Add ellites
        NumEllites = self.Config["NumEllites"]
        EllitesIndex = np.argpartition(Loss, NumEllites)[0:NumEllites].astype(int)
        self.KPopRes[CurrentGen]["EllitesIndex"] = EllitesIndex
        
        #---------- Fitness Calculation ----------
        # Calculate fitness probability for Roulette Wheel Selection.
        
        maxnorm = np.amax(Loss)
        normobj = maxnorm-Loss + 1     # The lowest obj has highest fitness. +1 to avoid 0.
        prob = normobj/np.sum(normobj)
        self.KPopRes[CurrentGen]["Prob"] = prob
        cumprob = np.cumsum(prob)
        #--------------------------------------
        
        #---------- Parents Selection ----------
        ParantSize = self.ParantSize
        ParentsIndex = np.empty(ParantSize)  # Create empty parents
        # Fill with ellite first.
        ParentsIndex[0:NumEllites] = EllitesIndex
        ## Then fill the rest by wheel withdrawing.
        for k in range(NumEllites, ParantSize):
            ParentsIndex[k] = np.searchsorted(cumprob,np.random.random())
        ## From the selected parents, we further randomly choose those who actually reproduce offsprings
        ef_par_list = np.array([False]*ParantSize)
        NumEffParents = 0
        while NumEffParents == 0:   # has to at least 1 parents to be selected
            for k in range(0, ParantSize):
                if np.random.random() <= self.Config["CrossProb"]:
                    ef_par_list[k] = True
                    NumEffParents += 1
        ## Effective parents
        EffParentsIndex = ParentsIndex[ef_par_list] 
        EffParents = self.Pop[CurrentGen][EffParentsIndex,:]
        self.EffLoss = Loss[EffParentsIndex]    # Record this so we don't re-evaluate them again for deterministic mode.
        self.NumEffParents = NumEffParents
        
        #---------- Children Formation ----------
        # Uniform crossover
        def UniformCrossover(parent1, parent2):
            child = np.zeros(NumPar)
            from1 = np.random.randint(0, 2, size = NumPar) == 0
            child[from1] = parent1[from1]
            child[~from1] = parent2[~from1]
            return child
        
        def Mutation(child):
            MutProb = self.Config["MutProb"]
            mut = np.random.binomial(n = 1, p = MutProb, size = NumPar) == 1
            MutSample_MC = self.MCSample(np.zeros((1,NumPar)))
            child[mut] = MutSample_MC.flatten()[mut]    # Since MutSample_MC.shape = (1, NumPar).
            return child
        
        def Mutation_Middle(child, p1, p2):
            MutProb = self.Config["MutProb"]
            MutSample_MC = self.MCSample(np.zeros((1,NumPar))).flatten() # Since MutSample_MC.shape = (1, NumPar).
            for i in range(self.NumPar):                           
                rnd = np.random.random()
                if rnd < MutProb:   
                    if p1[i] < p2[i]:
                        child[i] = p1[i]+np.random.random()*(p2[i] - p1[i])  
                    elif p1[i] > p2[i]:
                        child[i] = p2[i] + np.random.random()*(p1[i] - p2[i])
                    else:
                        child[i] = MutSample_MC[i] 
            return child
        
        # New generation
        PopNew = np.empty((PopSize, self.NumPar))
        ## First, fill with those selected parents without any modification
        PopNew[:ParantSize,:] = EffParents
        ## Then, fill the rest with crossover and mutation process
        for k in range(ParantSize, PopSize, 2):
            rn1 = np.random.randint(0, NumEffParents)
            rn2 = np.random.randint(0, NumEffParents)
            parent1 = EffParents[rn1]
            parent2 = EffParents[rn2]
            child1 = UniformCrossover(parent1, parent2)
            child2 = UniformCrossover(parent1, parent2)
            child1 = Mutation(child1)
            child2 = Mutation_Middle(child2)
            PopNew[k,:] = child1
            PopNew[k+1,:] = child2
            
        self.Pop[CurrentGen+1] = PopNew
        #---------------------------------------------------------------------------------------

        #---------- Prepare For Next Gen ----------
        # Open store space for next generation.
        self.KPopRes[CurrentGen+1] = {}
        #------------------------------------------
        return None
    
    def dropRecord(self):
        """Drop historical populations. However, we still keep historical tracking for all results.
        """
        if self.Config["DropRecord"] and (self.CurrentGen-1) >= 0:
            # To make sure self.CurrentGen-1 exsit. 
            # When program shoutdown (Continue = Ture), we might encounter Pop[self.CurrentGen-1] has been deleted.
            try:    
                del self.Pop[self.CurrentGen-1]
            except:
                pass
            
    def autoSave(self):
        """Auto save a snapshot of current KGCA process in case any model break down.
        """
        CaliWD = self.CaliWD
        Snapshot = self.__dict__
        with open(os.path.join(CaliWD, "AutoSave.pickle"), 'wb') as outfile:
            pickle.dump(Snapshot, outfile)#, protocol=pickle.HIGHEST_PROTOCOL)
            # About protocol: https://stackoverflow.com/questions/23582489/python-pickle-protocol-choice
        return None
    
    def run(self, InitialPop = None):
        logger.info("Start KGCA......")
        #----- Setting timer
        self.start_time = time.monotonic()
        self.elapsed_time = 0
        
        SamplingMethod = self.Config["SamplingMethod"]
        MaxGen = self.Config["MaxGen"]
        AutoSave = self.Config["AutoSave"]
        
        #----- Initialize Pop[0]
        # If it is a continuous run from previous shutdown work, we don't need initialization.
        if self.Continue is not True:
            self.initialize(SamplingMethod, InitialPop)
        else:
            logger.info("Continue from Gen {}.".format(self.CurrentGen))
            print("Do you want to extend the MaxGen {}? [y/n/exit]".format(MaxGen))
            ans1 = input()
            if ans1 == "y":
                print("Enter the new MaxGen.")
                ans2 = int(input())
                self.Config["MaxGen"] = ans2
                MaxGen = self.Config["MaxGen"]
                if self.CurrentGen > MaxGen:
                    self.Pop[self.CurrentGen] = self.ForExtendRun["Pop"]
                    self.KPopRes[self.CurrentGen] = self.ForExtendRun["KPopRes"]
                elif ans1 == "exit":
                    print("Press to exit.")
                    input()
                    quit()
            
        self.HistoryResult = {}   
        # Run the loop until reach maximum generation. (Can add convergent termination critiria in the future.)
        while self.CurrentGen <= MaxGen:
            self.nextGen()      # GA process
            self.dropRecord()   # Delete previou generation's Pop if DropRecord = True
            
            #----- Extract solutions
            self.Result = {}
            self.Result["Gen"] = self.CurrentGen
            self.Result["GlobalOptimum"] = {}
            self.Result["GlobalOptimum"]["Loss"] = self.Best["Loss"][self.CurrentGen]
            self.Result["GlobalOptimum"]["Index"] = int(self.Best["Index"][self.CurrentGen])
            self.Result["GlobalOptimum"]["Solutions"] = self.scale(self.Pop[self.CurrentGen][self.Result["GlobalOptimum"]["Index"]])
            self.Result["Loss"] = self.KPopRes[self.CurrentGen]["EllitesLoss"]
            self.Result["Index"] = self.KPopRes[self.CurrentGen]["EllitesIndex"].astype(int)
            self.Result["Solutions"] = self.KPopRes[self.CurrentGen]["Ellites"]     # Already scaled.  
            self.HistoryResult[self.CurrentGen] = self.Result 
            #----- Auto save
            if AutoSave:        # If Autosave is True, a model snapshot (pickle file) will be saved at CaliWD.
                self.autoSave()
                
            #----- Print output    
            if self.CurrentGen%self.Config["Printlevel"] == 0:
                pg = int(self.CurrentGen/(MaxGen/10))
                ProgressBar = "#"*pg+"-"*(10-pg)        # Form progressice bar ###-------
                logger.info("{:4d}/{:4d}   |{}|.".format(self.CurrentGen, MaxGen, ProgressBar))
                if self.Config["Plot"]:                 # Plot Loss of all SPs. To visualize the convergence.
                    self.plotProgress()
                logger.info(Dict2String(self.Result))
            #----- Next generation
            logger.info("Complete Gen {}/{}.".format(self.CurrentGen, self.Config["MaxGen"]))
            
            self.CurrentGen += 1 
                   
        #----- Delete Pop with gen index = (MaxGen+1 -1)
        self.ForExtendRun = {"Pop": self.Pop[self.CurrentGen], "KPopRes": self.KPopRes[self.CurrentGen] }
        del self.Pop[self.CurrentGen]   
        del self.KPopRes[self.CurrentGen] 
        
        #----- Count duration.
        elapsed_time = time.monotonic() - self.start_time
        self.elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        logger.info("Done! [{}]".format(self.elapsed_time))
        logger.info("Report:\n" + Dict2String(self.Result))
        
        #----- Output Report txt file.
        with open(os.path.join(self.CaliWD, "Report_KGCA_" + self.__name__ + ".txt"), "w") as text_file:
            text_file.write(Dict2String(self.Result))
            text_file.write("\n=====================================================")
            text_file.write("Elapsed time:\n{}".format(self.elapsed_time))
            text_file.write("\n=====================================================")
            text_file.write("\nKGCA user input Config:\n")
            text_file.write(Dict2String(self.Config))
            
        if AutoSave:        # If Autosave is True, a model snapshot (pickle file) will be saved at CaliWD.
            self.autoSave()
        
    def plotProgress(self, Save = True):
        """Plot KGCA progress to visualize the convergence. 
        """        
        elapsed_time = time.monotonic() - self.start_time
        self.elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        
        fig, ax = plt.subplots()
        # Plot scatter points for ellites in each cluster.
        for gen in range(self.CurrentGen+1):
            EllitesLoss = self.KPopRes[gen]["EllitesLoss"]
            ax.plot([gen]*len(EllitesLoss), EllitesLoss, "+", color='gray')      
        # Plot global optimum.     
        x = np.arange(0, self.CurrentGen+1)   
        loss = self.Best["Loss"][:self.CurrentGen+1]
        ax.plot(x, loss, label = "Best \n Elapsed_time: [{}]".format(self.elapsed_time), linewidth = 2, color = "black")        
        ax.set_title(self.__name__)
        ax.set_xlim([0, self.Config["MaxGen"]])
        ax.set_ylim([0, loss[0]*1.1])
        ax.set_xlabel("Generation")
        ax.set_ylabel("Loss (Minimun = 0)")
        ax.legend()
        if Save:
            filename = os.path.join(self.CaliWD, "Loss_" + self.__name__ + ".png")
            fig.savefig(filename)
        plt.show()
        
    def plotElbow(self):
        CurrentGen = self.CurrentGen
        KClusterMin = self.Config["KClusterMin"] #- 1 # See #-----Kmeans
        KClusterMax = self.Config["KClusterMax"]
        KDistortions = self.KPopRes[CurrentGen]["KDistortions"]
        KExplainedVar = self.KPopRes[CurrentGen]["KExplainedVar"]
        # KLeastImproveRate = self.Config["KLeastImproveRate"]
        # KExplainedVarThres = self.Config["KExplainedVarThres"]
        SelectedK = self.KPopRes[CurrentGen]["SelectedK"]
        # Kby = self.KPopRes[CurrentGen]["Kby"]
        # Elbow plot
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot(range(KClusterMin, KClusterMin+len(KDistortions)), KDistortions, marker='o', markersize=5, c = "blue")
        ax2.plot(range(KClusterMin, KClusterMin+len(KDistortions)), KExplainedVar, marker='o', markersize=5, c = "orange")
        ax.set_title(self.__name__ + " (Gen {})".format(CurrentGen))
        ax.axvline(x=SelectedK, color = "red")
        # if Kby == "ImproveRate":
        #     ax.scatter(SelectedK, KDistortions[-2], s=100, 
        #             facecolors='none', edgecolors='r', label = "  Selected K \n(Rate = {})".format(KLeastImproveRate))
        #     ax.legend(loc = "center right")
        # elif Kby == "ExplainedVar":
        #     ax2.scatter(SelectedK, KExplainedVar[-1], s=100, 
        #             facecolors='none', edgecolors='r', label = "  Selected K \n(Thres = {})".format(KExplainedVarThres))
        #     ax2.legend(loc = "center right")
        #     ax2.axhline(y=KExplainedVarThres, ls = "--", lw = 0.5, c = "grey")
        # if Kby == "KMax":
        #     ax.scatter(SelectedK, KDistortions[-1], s=100, 
        #             facecolors='none', edgecolors='r', label = "  Selected K \n(Reach K Max)".format(KLeastImproveRate))
        #     ax.legend(loc = "center right")
        ax.set_xlabel("Number of clusters (Max K = {})".format(KClusterMax))
        ax.set_ylabel("Distortion (within cluster sum of squares")    
        ax2.set_ylabel("Explained Variance")
        ax2.set_ylim([0,1])
        
        plt.show()
        
        
    def plotSilhouetteAvg(self):
        CurrentGen = self.CurrentGen
        SelectedK = self.KPopRes[CurrentGen]["SelectedK"]
        KClusterMin = self.Config["KClusterMin"]
        KClusterMax = self.Config["KClusterMax"]
        SilhouetteAvg = self.KPopRes[CurrentGen]["SilhouetteAvg"]
        fig, ax = plt.subplots()
        ax.plot(range(KClusterMin, KClusterMin+len(SilhouetteAvg)), SilhouetteAvg, marker='o', markersize=5, c = "blue")
        ax.axvline(x=SelectedK, color = "red")
        ax.set_ylim([-1,1])
        ax.set_title(self.__name__ + " (Gen {})".format(CurrentGen))
        ax.set_xlabel("Number of clusters (Max K = {})".format(KClusterMax))
        ax.set_ylabel("Silhouette Averge")    
        plt.show()