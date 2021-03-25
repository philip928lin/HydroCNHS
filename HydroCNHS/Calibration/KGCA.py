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
          "LocalSearchIter":    5,      # Number of iterations for each individual for its local search.
          "LocalSearchIta":     0.2,    # Initial local search initial step size. 
          "MaxGen":             100,    # Maximum generation.
          "SamplingMethod":     "LHC",  # MC: Monte Carlo sampling method. LHC: Latin Hyper Cube. (for initial pop)
          "FeasibleTolRate":    1.2,    # A dynamic criteria according to the best loss. Should be >= 1.
          "FeasibleThres":      0.3,    # A fix threshold for loss value.
          "MutProb":            0.3,    # Mutation probability.
          "KClusterMin":        2,      # >= 1
          "KClusterMax":        10,     # Must be smaller than PopSize/2. Otherwise, you will more likely to encounter error. 
          "KInterval":          10,     # An interval to rerun Kmeans clustering.
          "DropRecord":         True,   # Population record will be dropped. However, ALL simulated results will still be kept. 
          "ParalCores":         2/None, # This will replace system config.
          "AutoSave":           True,   # Automatically save a model snapshot after each generation.
          "Printlevel":         10,     # Print out level. e.g. Every ten generations.
          "Plot":               True    # Plot loss and cluster number selection with Printlevel frequency.
          }
"""

class KGCA(object):
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
    
    def LocalSearch(self, pop, loss, i):
        """Local search along the axises.

        Args:
            pop (Array): Individual.
            loss (float): Loss value of the individual.
            i (int): Index of pop in Pop.

        Returns:
            tuple: (pop, LocalLoss)
        """
        # See (Poikolainen et al., 2015) DOI: 10.1016/j.ins.2014.11.026 for details.
        ita = self.ita[i]
        # Generate local search samples.
        Orgpop = pop
        pop = deepcopy(pop)
        NumPar = self.NumPar
        LocalLoss = loss
        for k in range(NumPar):
            temp = deepcopy(pop)
            localpop = deepcopy(temp)
            localpop[k] = localpop[k] - ita
            if localpop[k] >=1:  localpop[k] = 1
            if localpop[k] <=0:  localpop[k] = 0
            Scaledlocalpop = self.scale(localpop)
            localloss = self.LossFunc(Scaledlocalpop, self.Formatter, (self.CaliWD, self.CurrentGen, "", i))
            if localloss < LocalLoss:
                pop = localpop
                LocalLoss = localloss
            elif localloss > LocalLoss:
                localpop = deepcopy(temp)
                localpop[k] = localpop[k] + 0.5 * ita
                if localpop[k] >=1:  localpop[k] = 1
                if localpop[k] <=0:  localpop[k] = 0
                Scaledlocalpop = self.scale(localpop)
                localloss = self.LossFunc(Scaledlocalpop, self.Formatter, (self.CaliWD, self.CurrentGen, "", i))
                if localloss < LocalLoss:
                    pop = localpop
                    LocalLoss = localloss
            if all(Orgpop == pop):
                ita = 0.5*ita
        self.ita[i] = ita
        return (pop, LocalLoss)
      
    def initialize(self, SamplingMethod = "LHC", InitialPop = None):
        """Initialize population members and storage spaces (KPopRes) for generation 0.

        Args:
            SamplingMethod (str, optional): Selected method for generate initial population members. MC or LHC.
            InitialPop (dict, optional): User-provided initial Pop[0] = [PopSize, NumPar] (2D array). Note that Pop[0] has to be scaled to [0,1] Defaults to None.
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
            # Initialize parameters with user inputs.
            self.Pop[0] = InitialPop

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
        LossParel = Parallel(n_jobs = ParalCores, verbose = ParalVerbose) \
                           ( delayed(LossFunc)\
                             (ScaledPop[k], Formatter, (self.CaliWD, CurrentGen, "", k)) \
                              for k in range(PopSize) )  # Still go through entire Pop including ellites.
        # Get results
        self.KPopRes[CurrentGen]["Loss"] = np.array(LossParel[0:PopSize])
        #--------------------------------------
        
        #---------- Initial Local Search ----------
        self.ita = {i: self.Config["LocalSearchIta"] for i in range(PopSize)}
        if self.CurrentGen == 0:
            for iter in range(self.Config["LocalSearchIter"]):
                logger.info("Local search iteration {}/{}.".format(iter+1, self.Config["LocalSearchIter"]))
                Pop = self.Pop[CurrentGen]
                Loss = self.KPopRes[CurrentGen]["Loss"]
                LocalPopAndLoss = Parallel(n_jobs = ParalCores, verbose = ParalVerbose) \
                                        ( delayed(self.LocalSearch)\
                                        (Pop[k], Loss[k], k) \
                                        for k in range(PopSize) )  # Still go through entire Pop including ellites.
                # Get results. Replace original initial Pop
                LocalLoss = []
                for k in range(PopSize):
                    self.Pop[CurrentGen][k] = np.array(LocalPopAndLoss[k][0])
                    LocalLoss.append(LocalPopAndLoss[k][1])
                self.KPopRes[CurrentGen]["Loss"] = np.array(LocalLoss)
                logger.debug("Local search Loss: \n{}".format(self.KPopRes[CurrentGen]["Loss"]))
        #------------------------------------------
        
        
        #---------- Feasibility ----------
        # We define 1: feasible sol and 0: infeasible sol.

        # Record the current global optimum. 
        BestIndex = np.argmin(self.KPopRes[CurrentGen]["Loss"])
        Best = self.KPopRes[CurrentGen]["Loss"][BestIndex]
        self.Best["Loss"][CurrentGen] = Best            # Array
        self.Best["Index"][CurrentGen] = BestIndex      # Array
        
        
        # Determine the feasibility.
        TolRate = self.Config["FeasibleTolRate"]        # Should be >= 1
        try:
            Thres = self.Config["FeasibleThres"]     
        except:
            Thres = 0   # Calibration minimum.         
        Feasibility = np.zeros(PopSize)
        Criteria = max([Best*TolRate, Thres])
        Feasibility[self.KPopRes[CurrentGen]["Loss"] <= Criteria] = 1   # Only valid when lower bound is zero
        self.KPopRes[CurrentGen]["Feasibility"] = Feasibility.astype(int)
        #---------------------------------
        
        
        #---------- Kmeans GA - Select eligible subPop & Run Kmeans clustering ----------
        Feasibility = self.KPopRes[CurrentGen]["Feasibility"]
        Loss = self.KPopRes[CurrentGen]["Loss"]
        NumFeaSols = np.sum(Feasibility)

        #Feasible solutions' indexes.
        Feasibility = self.KPopRes[CurrentGen]["Feasibility"]
        Loss = self.KPopRes[CurrentGen]["Loss"]
        SubPopSizeThres = int(PopSize/2)
        # We set the criteria as PopSize/2 for now, which can be switch to dynamically updated according to CurrentGen.
        # Select subPop for clustering => Max(#Feasible Sol, PopSize/2)
        if np.sum(Feasibility) >= SubPopSizeThres:
            KPopIndex = np.where(Feasibility == 1)[0]   # Take out 1d array of indexes.
        else:
            KPopIndex = np.argpartition(Loss, SubPopSizeThres)[:SubPopSizeThres].astype(int)          # return n smallest Loss index.
        KPop = self.Pop[CurrentGen][KPopIndex, :]
        self.KPopRes[CurrentGen]["KPopIndex"] = KPopIndex
        
        
        # KPopIndex = np.where(Feasibility == 1)[0]   # Take out 1d array of indexes.
        # KPop = self.Pop[CurrentGen][KPopIndex, :]   # Every pop is [0, 1]
        # self.KPopRes[CurrentGen]["KPopIndex"] = KPopIndex
        
        #----- Kmeans clustering
        ## Check eligibility of KClusterMax
        if CurrentGen%self.Config["KInterval"] == 0 or CurrentGen == self.Config["MaxGen"]:
            if self.Config["KClusterMax"] > NumFeaSols:
                logger.warning("Number of feasible solutions is less than KClusterMax. We reset KClusterMax to {} for Gen {}.".format(NumFeaSols, self.Config["KClusterMax"]))
            KClusterMax = int(min(self.Config["KClusterMax"], len(KPop)/2))
            KClusterMin = self.Config["KClusterMin"]
            ParWeight = self.Inputs.get("ParWeight")     # Weights for each parameter. Default None. 
            
            KmeansModel = {}
            KDistortions = []
            KExplainedVar = []
            SilhouetteAvg = []
            SSE = np.sum(np.var(KPop, axis = 0))*KPop.shape[0]
            for k in range(KClusterMin, KClusterMax+1):
                km = KMeans(n_clusters = k, random_state=0).fit(KPop, ParWeight)
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
                    silhouette_avg = silhouette_score(KPop, cluster_labels)
                    SilhouetteAvg.append(silhouette_avg)

            # Store records.
            MaxSilhouetteAvg = max(SilhouetteAvg)
            self.KPopRes[CurrentGen]["SelectedK"] = SilhouetteAvg.index(MaxSilhouetteAvg) + KClusterMin
            self.KPopRes[CurrentGen]["SilhouetteAvg"] = SilhouetteAvg
            self.KPopRes[CurrentGen]["KDistortions"] = KDistortions
            self.KPopRes[CurrentGen]["KExplainedVar"] = KExplainedVar
            KM = KmeansModel[self.KPopRes[CurrentGen]["SelectedK"]]
            self.KM = KM
            self.KPopRes[CurrentGen]["Centers"] = self.scale(KM.cluster_centers_)
        else:
            # Retrieve data from last generation.
            self.KPopRes[CurrentGen]["SelectedK"] = self.KPopRes[CurrentGen-1]["SelectedK"]
            self.KPopRes[CurrentGen]["SilhouetteAvg"] = self.KPopRes[CurrentGen-1]["SilhouetteAvg"]
            self.KPopRes[CurrentGen]["KDistortions"] = self.KPopRes[CurrentGen-1]["KDistortions"]
            self.KPopRes[CurrentGen]["KExplainedVar"] = self.KPopRes[CurrentGen-1]["KExplainedVar"]
            KM = self.KM
            self.KPopRes[CurrentGen]["Centers"] = self.KPopRes[CurrentGen-1]["Centers"]
        
        # Assign all pop according to KM model. Therefore those infeasible solutions will still participate in the tournament.
        self.KPopRes[CurrentGen]["PopLabels"] = KM.fit_predict(self.Pop[CurrentGen])
        if self.Config["Plot"] and self.CurrentGen%self.Config["Printlevel"] == 0:
            self.plotElbow()
            self.plotSilhouetteAvg()
        #---------------------------------------------------------------------------------
        
        #---------- Kmeans GA - Select parents and Generate children for each cluster ----------
        Loss = self.KPopRes[CurrentGen]["Loss"]
        KLabels = self.KPopRes[CurrentGen]["PopLabels"]
        Pop = self.Pop[CurrentGen]
        SelectedK = self.KPopRes[CurrentGen]["SelectedK"]
        
        # Select ellite and calculate rank of each cluster according to the best fitness individual in the cluster.
        KElliteIndex = {}
        self.KPopRes[CurrentGen]["Ellites"] = np.zeros((SelectedK, NumPar))
        self.KPopRes[CurrentGen]["EllitesIndex"] = np.zeros(SelectedK)
        self.KPopRes[CurrentGen]["EllitesLoss"] = np.zeros(SelectedK)
        self.KPopRes[CurrentGen]["EllitesProb"] = np.zeros(SelectedK)
        self.KIndex = {}
        for k in range(SelectedK):
            KIndex = np.where(KLabels == k)[0]
            self.KIndex[k] = KIndex
            Loss_k = Loss[KIndex]
            KElliteNum = 1  # We only take 1 ellite for each cluster.
            if len(Loss_k) <= KElliteNum:     # If we only have 1 choice.
                try:
                    KElliteIndex[k] = KIndex[0]
                except:
                    logger.error("Some clusters have no assigned members. Try to lower KClusterMin or higher KLeastImproveRate.")
                    raise ValueError("Empty list.")
            else: 
                # return 1 smallest Loss.
                KElliteIndex[k] = KIndex[np.argpartition(Loss_k, KElliteNum)[0].astype(int)] 
        
            # Store ellite of each cluster.
            self.KPopRes[CurrentGen]["Ellites"][k] = self.scale(Pop[int(KElliteIndex[k])])
            self.KPopRes[CurrentGen]["EllitesIndex"][k] = int(KElliteIndex[k])
            self.KPopRes[CurrentGen]["EllitesLoss"][k] = Loss[int(KElliteIndex[k])]
        
        # Calculate group prob
        ## Feasibility based EllitesProb assignment.
        EllitesLoss = self.KPopRes[CurrentGen]["EllitesLoss"]    
        EllitesLoss = max(EllitesLoss) - EllitesLoss +1 # To avoid dividing 0.
        KProb = EllitesLoss/np.sum(EllitesLoss)
        ## Assign feasible cluster with equal prob.
        Criteria = max([Best*TolRate, Thres])
        feasi = self.KPopRes[CurrentGen]["EllitesLoss"] <= Criteria    # Only valid when lower bound is zero
        KProb[feasi] = np.mean(KProb[feasi])
        self.KPopRes[CurrentGen]["EllitesProb"] = KProb
        
        # ## Rank based EllitesProb assignment.
        # Rank = SelectedK+1 - rankdata(self.KPopRes[CurrentGen]["EllitesLoss"]) # Lower value higher rank (min = 1)
        # KProb = Rank/np.sum(Rank)
        # self.KPopRes[CurrentGen]["EllitesRank"] = Rank
        # self.KPopRes[CurrentGen]["EllitesProb"] = KProb
        
        def RouletteWheelSelection(KProb):
            rn = np.random.uniform(0,1)
            acc1 = 0; acc2 = 0
            for i, v in enumerate(KProb):
                acc2 += v
                if rn >= acc1 and rn < acc2:
                    return i
                acc1 += v
                
        # Uniform crossover
        def UniformCrossover(parent1, parent2):
            child = np.zeros(NumPar)
            from1 = np.random.randint(0, 2, size = NumPar) == 0
            child[from1] = parent1[from1]
            child[~from1] = parent2[~from1]
            return child
        
        def Mutation(child):
            mut = np.random.binomial(n = 1, p = MutProb*MutPartition[1], size = NumPar) == 1
            MutSample_MC = self.MCSample(np.zeros((1,NumPar)))
            child[mut] = MutSample_MC.flatten()[mut]    # Since MutSample_MC.shape = (1, NumPar).
            return child
        # def Mutation(child):
        #     # Half mutate from MC sampling, half from local pertabation.
        #     rn = np.random.binomial(n = 1, p = 0.5, size = NumPar)
        #     mut1 = np.random.binomial(n = 1, p = MutProb*MutPartition[1], size = NumPar)*rn == 1
        #     mut2 = np.random.binomial(n = 1, p = MutProb*MutPartition[1], size = NumPar)*(1-rn) == 1
        #     MutSample_MC = self.MCSample(np.zeros((1,NumPar)))
        #     WithinGroup = np.array(  [truncnorm.rvs(0,1, loc=m, scale=0.1) for m in child]  )
        #     child[mut1] = MutSample_MC.flatten()[mut1]    
        #     child[mut2] = WithinGroup.flatten()[mut2]    
        #     return child
        
        MutProb = self.Config["MutProb"]
        MutPartition = (0.3, 0.7)
        if SelectedK == 1:
            MutPartition = (0, 1)   # No inter clusters mutation.
        Pop = self.Pop[CurrentGen] 
        self.Pop[CurrentGen+1] = np.zeros((PopSize, NumPar))
        for p in range(int(PopSize/2)):
            k = RouletteWheelSelection(KProb)
            MutRn = np.random.uniform(0,1)
            if MutRn <= MutProb*MutPartition[0]:
                # Mutation form 1 self sampling => reinforce local area & cross cluster.
                BestpopInK = Pop[int(KElliteIndex[k])]
                # Use truncated normal to sample around in-cluster best solution.
                child1 = np.array(  [truncnorm.rvs(0,1, loc=m, scale=0.1) for m in BestpopInK]  )
                # Crossover with other ellite in other group.
                k2 = np.random.choice( [j for j in range(SelectedK) if j != k] )
                parent2 = Pop[int(KElliteIndex[k2])]
                child2 = UniformCrossover(BestpopInK, parent2)[0]
            else:
                KIndex = self.KIndex[k]
                pair1 = [np.random.choice(KIndex), np.random.choice(KIndex)]
                parent1 = Pop[ pair1[  np.argmin( [Loss[pair1[0]], Loss[pair1[1]]] )  ]]
                pair2 = [np.random.choice(KIndex), np.random.choice(KIndex)]
                parent2 = Pop[ pair2[  np.argmin( [Loss[pair2[0]], Loss[pair2[1]]] )  ]]
                child1 = UniformCrossover(parent1, parent2)
                child2 = UniformCrossover(parent1, parent2)
                child1 = Mutation(child1)
                child2 = Mutation(child2)
            self.Pop[CurrentGen+1][2*p] = child1
            self.Pop[CurrentGen+1][2*p+1] = child2
        
        for k in range(SelectedK):
            # Replace top k pop with ellites from each cluster if it is feasible. 
            if self.KPopRes[CurrentGen]["EllitesLoss"][k] <= Criteria:
                self.Pop[CurrentGen+1][k] = Pop[int(KElliteIndex[k])]
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
        Snapshot = self.__dict__.copy()
        with open(os.path.join(CaliWD, "AutoSave.pickle"), 'wb') as outfile:
            pickle.dump(Snapshot, outfile)#, protocol=pickle.HIGHEST_PROTOCOL)
            # About protocol: https://stackoverflow.com/questions/23582489/python-pickle-protocol-choice
        return None
    
    def run(self, InitialPop = None):
        logger.info("Start KGCA......")
        #----- Setting timer
        start_time = time.monotonic()
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
        del self.Pop[self.CurrentGen]   
        del self.KPopRes[self.CurrentGen] 
        
        

        
        #----- Count duration.
        elapsed_time = time.monotonic() - start_time
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
        fig, ax = plt.subplots()
        # Plot scatter points for ellites in each cluster.
        for gen in range(self.CurrentGen+1):
            EllitesLoss = self.KPopRes[gen]["EllitesLoss"]
            ax.plot([gen]*len(EllitesLoss), EllitesLoss, "+", color='gray')      
        # Plot global optimum.     
        x = np.arange(0, self.CurrentGen+1)   
        loss = self.Best["Loss"][:self.CurrentGen+1]
        ax.plot(x, loss, label = "Best", linewidth = 2, color = "black")        
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