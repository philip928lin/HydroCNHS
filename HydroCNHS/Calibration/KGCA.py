#%%
# Kmeans genetic algorithm (GA).
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# We generalized the code and add a mutation method call mutation_middle.
# However, we deactivate mutation_middle for DMC. This function helps convergence but restricts exploration in DMC case.
# 2021/02/25

from ..SystemConrol import loadConfig, Dict2String   # HydroCNHS module
from joblib import Parallel, delayed                 # For parallelization
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import logging
import time
from sklearn.cluster import KMeans
logger = logging.getLogger("HydroCNHS.KGCA") # Get logger 

r"""
Need to be added sometime.
Check function 
Force parallel in HydroCNHS to stop
Timeout (function) 
"""

r"""
Inputs = {"ParName":[], 
          "ParBound":[],    # [upper, low] or [4, 6, 9] Even for categorical type, it has to be numbers!
          "ParType":[],     # real or categorical
          #"ParWeight":[],  # An array with length equal to number of parameters.
          "WD":}   
          
Config = {#"NumSP":0,               # Number of sub-populations.
          "PopSize": 30,            # Population size. Must be even.
          "MaxGen": 100,            # Maximum generation.
          "SamplingMethod": "LHC",  # MC: Monte Carlo sampling method. LHC: Latin Hyper Cube. (for initial pop)
          "FeasibleTolRate": 1.2    # A dynamic criteria according to the best loss. Should be >= 1.
          "FeasibleThres": 0.3      # A fix threshold for loss value.
          #"NumEllite": 1,          # Ellite number for each SP. At least 1.
          "MutProb": 0.3,           # Mutation probability.
          "KClusterMin": 2,         # Must be at least 2. See #----- Kmeans clustering
          "KClusterMax": 10,        # Must be smaller than PopSize. 
          "KLeastImproveRate": 0.5, # Improving rate criteria for k cluster selection.
          "KExplainedVarThres": 0.8,# Total explained criteria for k cluster selection.
          "DropRecord": True,       # Population record will be dropped. However, ALL simulated results will remain. 
          "ParalCores": 2/None,     # This will overwrite system config.
          "AutoSave": True,         # Automatically save a model snapshot after each generation.
          "Printlevel": 10,         # Print out level. e.g. Every ten generations.
          "Plot": True              # Plot loss with Printlevel frequency.
          }
"""
r"""
Psuedo code for KmeansGA
Input KmeansGA Config setting 
(e.g. PopSize, NumPar, FeasibleTolRate, FeasibleThres, 
KClusterMin/Max, KLeastImproveRate, KExplainedVarThres)

# Intialization
Pop <= np.empty((PopSize, NumPar))
Pop <= Latin Hyper Cube sampling.
CurrentGen = 0

while CurrentGen <= MaxGen:
    #---------- Evaluation ----------
    Loss = np.empty((PopSize, NumPar))
    for p in range(PopSize):
        Loss[p] = LossFunc(Pop[p])
    
    #---------- Feasibility ----------
    Feasibility = np.zero(PopSize)
    Best = Min(Loss)
    Criteria = max([Best*FeasibleTolRate, FeasibleThres])
    Feasibility[Loss <= Criteria] = 1 
    
    #---------- SubPop Selection ----------
    if num of feasible solutions >= PopSize/2:
        SubPop = Pop[Feasibility == 1]
    else:
        SubPop = PopSize/2 of best Loss value.
    SubPop <= Scaled SubPop ([0,1])
     
    #---------- Kmeans: K selection ----------    
    for k in range(KClusterMin-1, KClusterMax):
        KMeans(n_clusters = k).fit(SubPop, ParWeight)
        if ExplainedVariance >= KExplainedVarThres:
            K = k
            break loop
        if ImproveRate < KLeastImproveRate:
            K = k-1
            break loop

    #---------- GA Evolution Process for Each SubPop ----------
    for sub in range(K):
        Select single ellite.
        Select parants through binary tournament. 
        Uniform Crossover and Mutation.
    Pop <= Collect all evolved SubPop and Add ellites of each sub-Pop.

    #---------- Prepare Next Iteration ----------
    CurrentGen += 1
"""

class KGCA(object):
    def __init__(self, LossFunc, Inputs, Config, Formatter = None, ContinueFile = None, Name = "Calibration"):
        """Diverse model calibrations (DMC) genetic algorithm (GA) object.

        Args:
            LossFunc (function): Loss function => LossFunc(pop, Formatter, SubWDInfo = None) and return loss, which has lower bound 0.
            Inputs (dict): Inputs dictionary, which can be generated by GA_Convertor. It contains ParName, ParBound, ParType, ParWeight, and WD.
            Config (dict): Config dictionary, which contains NumSP, PopSize, MaxGen, SamplingMethod, Tolerance, NumEllite, MutProb, DropRecord, ParalCores (optional), AutoSave, Printlevel, and Plot.
            Formatter (dict, optional): Formatter dictionary created by GA_Convertor. This will be further feed back to LossFunc for user to convert 1D array back to original format to run HydroCNHS. Defaults to None.
            ContinueFile (str, optional): AutoSave.pickle directory to continue the previous run. Defaults to None.
            Name (str, optional): Name of the DMCGA object, corresponding to the created sub-folder name. Defaults to "Calibration".
        """
               
        # Populate class attributions.
        self.LossFunc = LossFunc                # Loss function LossFunc(pop, Formatter, SubWDInfo = None) return loss, which has lower bound 0.
                                                    #pop is a parameter vector. 
                                                    #Formatter can be obtained from class GA_Convertor.
                                                    #SubWDInfo = (CaliWD, CurrentGen, sp, k).
                                                    #Lower bound of return value has to be 0.
        self.Inputs = Inputs                    # Input ductionary.
        self.Inputs["ParWeight"] = np.array(Inputs["ParWeight"])    # Make sure it is an array.
        self.Config = Config                    # Configuration for DMCGA.
        self.Formatter = Formatter              # Formatter is to convert 1D pop back into list of dataframe dictionaries for HydroCNHS simulation. (see class GA_Convertor) 
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
            # Generate sub population index list for later for-loop and code readibility.
            # For KmeansGA => Config["NumSP"] must be 0.
            #self.SPList = ["SP0"] + ["SP"+str(i+1) for i in range(Config["NumSP"])]
            
            self.CurrentGen = 0     # Populate initial counter for generation.
            
            # Initialize variables storage.
            self.Pop = {}           # Population of parameter set. Pop[gen][sp]: [k,s] (2D array); k is index of members, and s is index of parameters.
            self.PopRes = {}        # Simulation results of each members. PopRes[gen][sp][Loss/Dmin/Feasibility/SelfD]: 1D array with length of population size.
            self.KPopRes = {}       # Store Kmeans results
            
            #self.SPCentroid = {}    # Centroid of each SP. SPCentroid[gen][sp][Centroid/NormalizedCentroid]: 1D array with length of number of calibrated parameters.
            
            # Best loss value and index of corresponding member in Pop[gen][sp]. Best[Loss/Index][sp]: 1D array with length of MaxGen.
            self.Best = {"Loss":  np.empty(self.Config["MaxGen"]+1),    # +1 since including gen 0.
                         "Index": np.empty(self.Config["MaxGen"]+1)}
            
            # Calculate scales for parameter normalization.
            # We assume categorical type is still number kind list (e.g. [1,2,3,4] and scale = 4-1 = 3).  
            self.BoundScale = []
            for i, ty in enumerate(Inputs["ParType"]):
                if ty == "real":
                    self.BoundScale.append(Inputs["ParBound"][i][1] - Inputs["ParBound"][i][0])
                elif ty == "categorical":
                    self.BoundScale.append(np.max(Inputs["ParBound"][i]) - np.min(Inputs["ParBound"][i]))
            self.BoundScale = np.array(self.BoundScale)     # Store in an array type. 
            
            # Create calibration folder under WD
            self.__name__ = Name
            self.CaliWD = os.path.join(Inputs["WD"], self.__name__)
            # Create CaliWD directory
            if os.path.isdir(self.CaliWD) is not True:
                os.mkdir(self.CaliWD)
            else:
                logger.warning("\n[!!!Important!!!] Current calibration folder exists. Default to overwrite the folder!\n{}".format(self.CaliWD))
        #---------------------------------------

    def MCSample(self, pop, ParBound, ParType):
        """Generate samples using Monte Carlo method.

        Args:
            pop (Array): 2D array. [PopSize, NumPar]
            NumPar (int): Number of parameters.
            ParBound (list): List of bounds for each parameters.
            ParType (list): List of parameter types. ["real" or "categorical"]

        Returns:
            array: Populated pop array.
        """
        PopSize = pop.shape[0]      # pop = [PopSize, NumPar]
        NumPar = pop.shape[1]
        for i in range(NumPar):
            if ParType[i] == "real":
                pop[:,i] = np.random.uniform(ParBound[i][0], ParBound[i][1], size = PopSize)  
            elif ParType[i] == "categorical":
                pop[:,i] = np.random.choice(ParBound[i], size = PopSize)
        return pop 
    
    def LatinHyperCubeSample(self, pop, ParBound, ParType):
        """Generate samples using Latin Hyper Cube (LHC) method. However, if the parameter type is categorical, we use MCSample instead.

        Args:
            pop (Array): 2D array. [PopSize, NumPar]
            NumPar (int): Number of parameters.
            ParBound (list): List of bounds for each parameters.
            ParType (list): List of parameter types. ["real" or "categorical"]

        Returns:
            array: Populated pop array.
        """
        PopSize = pop.shape[0]      # pop = [PopSize, NumPar]
        NumPar = pop.shape[1]
        for i in range(NumPar):
            if ParType[i] == "real":
                d = 1.0 / PopSize
                temp = np.empty([PopSize])
                # Uniformly sample in each interval.
                for j in range(PopSize):
                    temp[j] = np.random.uniform(low=j * d, high=(j + 1) * d)
                # Shuffle to random order.
                np.random.shuffle(temp)
                # Scale [0,1] to its bound.
                pop[:,i] = temp*(ParBound[i][1] - ParBound[i][0]) + ParBound[i][0]
            elif ParType[i] == "categorical":
                pop[:,i] = np.random.choice(ParBound[i], size = PopSize)
        return pop 
        
    def initialize(self, SamplingMethod = "LHC", InitialPop = None):
        """Initialize population members and storage spaces (PopRes, SPCentroid) for each sub population for generation 0.

        Args:
            SamplingMethod (str, optional): Selected method for generate initial population members. MC or LHC.
            InitialPop (dict, optional): User-provided initial Pop[0]. InitialPop[sp]: [PopSize, NumPar] (2D array). Defaults to None.
        """
        PopSize = self.Config["PopSize"]
        NumPar = self.NumPar
        ParBound = self.Inputs["ParBound"]
        ParType =  self.Inputs["ParType"]

        # Initialize storage space for generation 0.
        if InitialPop is None:      # Initialize parameters according to selected sampling method.
            pop = np.zeros((PopSize, NumPar))  # Create 2D array population for single generation.
            if SamplingMethod == "MC":
                self.Pop[0] = self.MCSample(pop, ParBound, ParType)
            elif SamplingMethod == "LHC":
                self.Pop[0] = self.LatinHyperCubeSample(pop, ParBound, ParType)
        else:                       # Initialize parameters with user inputs.
            self.Pop[0] = InitialPop
        
        # Initialize storage space for each SP
        self.KPopRes[0] = {}
        
    def nextGen(self):
        """Complete all simulations and generate next generation of Pop.
        Detail procedure, please see Algorithm 1 in (Williams et al., 2020) https://doi.org/10.1016/j.envsoft.2020.104831.
        """
        LossFunc = self.LossFunc
        Formatter = self.Formatter
        CurrentGen = self.CurrentGen
        PopSize = self.Config["PopSize"]
        NumPar = self.NumPar
        # NumEllite = self.Config["NumEllite"]
        
        # Load parallelization setting (from user or system config)
        ParalCores = self.Config.get("ParalCores")
        if ParalCores is None:      # If user didn't specify ParalCores, then we will use default cores in the system config.
            ParalCores = self.SysConfig["Parallelization"]["Cores_DMCGA"]
        ParalVerbose = self.SysConfig["Parallelization"]["verbose"]         # Joblib print out setting.
        
        #---------- Evaluation (Min) ----------
        # Note: Since HydroCHNS is a stochastic model, we will re-simulate ellites!!
        # In future, we can have an option for this (re-simulate ellites) to further enhance computational efficiency.
        # Evalute Loss function
        # LossFunc(pop, Formatter, SubWDInfo = None) and return loss, which has lower bound 0.
        LossParel = Parallel(n_jobs = ParalCores, verbose = ParalVerbose) \
                           ( delayed(LossFunc)\
                             (self.Pop[CurrentGen][k], Formatter, (self.CaliWD, CurrentGen, "", k)) \
                              for k in range(PopSize) )  # Still go through entire Pop including ellites.
        # Get results
        self.KPopRes[CurrentGen]["Loss"] = np.array(LossParel[0:PopSize])
        #--------------------------------------
        
        #---------- Feasibility ----------
        # We define 1: feasible sol and 0: infeasible sol.

        # Record the current global optimum. 
        BestIndex = np.argmin(self.KPopRes[CurrentGen]["Loss"])
        Best = self.KPopRes[CurrentGen]["Loss"][BestIndex]
        self.Best["Loss"][CurrentGen] = Best            # Array
        self.Best["Index"][CurrentGen] = BestIndex      # Array
        
        
        # Determine the feasibility.
        TolRate = self.Config["FeasibleTolRate"]        # Should be >= 1
        Thres = self.Config["FeasibleThres"]              
        Feasibility = np.zeros(PopSize)
        Criteria = max([Best*TolRate, Thres])
        Feasibility[self.KPopRes[CurrentGen]["Loss"] <= Criteria] = 1   # Only valid when lower bound is zero
        self.KPopRes[CurrentGen]["Feasibility"] = Feasibility.astype(int)
        #---------------------------------
        
        #---------- Kmeans GA - Select eligible subPop & Run Kmeans clustering ----------
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
        
        # Normalize for computing distance
        Nor_KPop = np.divide(KPop, self.BoundScale.reshape(1, NumPar))
        
        #----- Kmeans clustering
        KClusterMin = self.Config["KClusterMin"]
        KClusterMin = KClusterMin - 1   # So we can select K = KClusterMin. Selecting criteria is the different in slope! 
        KClusterMax = self.Config["KClusterMax"]
        KLeastImproveRate = self.Config["KLeastImproveRate"]
        KExplainedVarThres = self.Config["KExplainedVarThres"]
        ParWeight = self.Inputs.get("ParWeight")     # Weights for each parameter. Default None. 
        KDistortions = []
        KExplainedVar = []
        KdDistortions = []
        KStore = [0, 0]     # Temporary store last 2 kmeans model. 
        SelectedK = KClusterMax
        SSE = np.sum(np.var(Nor_KPop, axis = 0))*Nor_KPop.shape[0]
        
        for k in range(KClusterMin, KClusterMax+1):
            KStore[0] = KStore[1]
            km = KMeans(n_clusters = k, random_state=0).fit(Nor_KPop, ParWeight)
            KStore[1] = km
            # inertia_: Sum of squared distances of samples to their closest cluster center.
            KDistortions.append(km.inertia_)
            KExplainedVar.append((SSE - KDistortions[-1])/SSE)
            if k >= KClusterMin+1:
                d_k = KDistortions[-1] - KDistortions[-2]
                KdDistortions.append(d_k)
                if KExplainedVar[-1] >= KExplainedVarThres:
                    SelectedK = k
                    self.KPopRes[CurrentGen]["Kby"] = "ExplainedVar"
                    print("Select k = {}, Explained Var = {}.".format(SelectedK, KExplainedVar[-1]))
                    break
            if len(KdDistortions) >= 2:
                if KdDistortions[-1]/KdDistortions[-2] < KLeastImproveRate:
                    SelectedK = k-1
                    self.KPopRes[CurrentGen]["Kby"] = "ImproveRate"
                    print("Select k = {}, KImproveRate = {}.".format(SelectedK, KdDistortions[-1]/KdDistortions[-2]))
                    break
            self.KPopRes[CurrentGen]["Kby"] = "KMax"
            
        if self.KPopRes[CurrentGen]["Kby"] == "ImproveRate":
            KM = KStore[0]      # Extract the final model for selected K.
        else:
            KM = KStore[1]
        self.KPopRes[CurrentGen]["SelectedK"] = SelectedK
        self.KPopRes[CurrentGen]["KDistortions"] = KDistortions
        self.KPopRes[CurrentGen]["KExplainedVar"] = KExplainedVar
        self.KPopRes[CurrentGen]["Centers"] = np.multiply(KM.cluster_centers_, self.BoundScale.reshape(1, NumPar)) 
        KLabels = np.empty(PopSize); KLabels[:] = np.nan
        KLabels[KPopIndex] = KM.labels_
        self.KPopRes[CurrentGen]["PopLabels"] = KLabels
        if self.Config["Plot"] and self.CurrentGen%self.Config["Printlevel"] == 0:
            self.plotElbow()
        #---------------------------------------------------------------------------------
        
        #---------- Kmeans GA - Select parents and Generate children for each cluster ----------
        Loss = self.KPopRes[CurrentGen]["Loss"]
        KLabels = self.KPopRes[CurrentGen]["PopLabels"]
        Pop = self.Pop[CurrentGen]
        MutProb = self.Config["MutProb"]
        
        KElliteIndex = {}
        KParentIndex = {}
        KChildren = {}
        # Make sure SubPopSize is even number and SubPopSize*SelectedK >= PopSize
        SubPopSize = int(PopSize/SelectedK) + 1
        if SubPopSize%2 != 0:
            SubPopSize += 1
            
        self.KPopRes[CurrentGen]["Ellites"] = np.zeros((SelectedK, NumPar))
        self.KPopRes[CurrentGen]["EllitesIndex"] = np.zeros(SelectedK)
        self.KPopRes[CurrentGen]["EllitesLoss"] = np.zeros(SelectedK)
        
        for k in range(SelectedK):
            KIndex = np.where(KLabels == k)[0]
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
                        
            # Parents selection by binary tournament
            KParentIndex[k] = np.zeros(SubPopSize)
            KChildren[k] = np.zeros((SubPopSize, NumPar))
            for i in range(SubPopSize):
                pair = [np.random.choice(KIndex), np.random.choice(KIndex)]
                KParentIndex[k][i] = pair[ np.argmin( [Loss[pair[0]],  
                                                       Loss[pair[1]]] ) ]
            
            # Uniform crossover
            def UniformCrossover(parent1, parent2):
                child = np.zeros(NumPar)
                from1 = np.random.randint(0, 2, size = NumPar) == 0
                child[from1] = parent1[from1]
                child[~from1] = parent2[~from1]
                return child
            
            def Mutation(child):
                mut = np.random.binomial(n = 1, p = MutProb, size = NumPar) == 1
                MutSample_MC = self.MCSample(np.zeros((1,NumPar)), self.Inputs["ParBound"], self.Inputs["ParType"])
                child[mut] = MutSample_MC.flatten()[mut]    # Since MutSample_MC.shape = (1, NumPar).
                return child
            
            for p in range(int(SubPopSize/2)):     # PopSize must be even.
                parent1 = Pop[ int(KParentIndex[k][2*p]) ]     
                parent2 = Pop[ int(KParentIndex[k][2*p+1]) ]   
                child1 = UniformCrossover(parent1, parent2)
                child2 = UniformCrossover(parent1, parent2)
                child1 = Mutation(child1)
                child2 = Mutation(child2)       
                KChildren[k][2*p] = child1
                KChildren[k][2*p+1] = child2
            # Replace first 1 pop with ellite. 
            KChildren[k][0] = Pop[int(KElliteIndex[k])]
            # Store ellite of each cluster.
            self.KPopRes[CurrentGen]["Ellites"][k] = Pop[int(KElliteIndex[k])]
            self.KPopRes[CurrentGen]["EllitesIndex"][k] = int(KElliteIndex[k])
            self.KPopRes[CurrentGen]["EllitesLoss"][k] = Loss[int(KElliteIndex[k])]
        
        # Fill KChildren into new gen of pop.
        self.Pop[CurrentGen+1] = {}
        self.Pop[CurrentGen+1] = np.zeros((PopSize, NumPar))
        for p in range(PopSize):
            self.Pop[CurrentGen+1][p] = KChildren[int(p%SelectedK)][int(p/SelectedK)]
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
        """Auto save a snapshot of current DMCGA process in case any model break down.
        """
        CaliWD = self.CaliWD
        Snapshot = self.__dict__.copy()
        with open(os.path.join(CaliWD, "AutoSave.pickle"), 'wb') as outfile:
            pickle.dump(Snapshot, outfile, protocol=pickle.HIGHEST_PROTOCOL)
            # About protocol: https://stackoverflow.com/questions/23582489/python-pickle-protocol-choice
        return None
    
    def run(self, InitialPop = None):
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
            
        # Run the loop until reach maximum generation. (Can add convergent termination critiria in the future.)
        while self.CurrentGen <= MaxGen:
            self.nextGen()      # GA process
            self.dropRecord()   # Delete previou generation's Pop if DropRecord = True
            
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
                    
            #----- Next generation
            self.CurrentGen += 1 
                            
                
        #----- Delete Pop with gen index = (MaxGen+1 -1)
        del self.Pop[self.CurrentGen]   
        del self.KPopRes[self.CurrentGen] 
        
        #----- Extract solutions
        self.Result = {}
        self.Result["GlobalOptimum"] = {}
        self.Result["GlobalOptimum"]["Loss"] = self.Best["Loss"][self.CurrentGen - 1]
        self.Result["GlobalOptimum"]["Index"] = int(self.Best["Index"][self.CurrentGen - 1])
        self.Result["GlobalOptimum"]["Solutions"] = self.Pop[self.CurrentGen - 1][self.Result["GlobalOptimum"]["Index"]]
        self.Result["Loss"] = self.KPopRes[self.CurrentGen - 1]["EllitesLoss"]
        self.Result["Index"] = self.KPopRes[self.CurrentGen - 1]["EllitesIndex"].astype(int)
        self.Result["Solutions"] = self.KPopRes[self.CurrentGen - 1]["Ellites"] 
        
        #----- Count duration.
        elapsed_time = time.monotonic() - start_time
        self.elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        logger.info("Done! [{}]".format(self.elapsed_time))
        logger.info("Report:\n" + Dict2String(self.Result))
        with open(os.path.join(self.CaliWD, "Report_KGCA_" + self.__name__ + ".txt"), "w") as text_file:
            text_file.write(Dict2String(self.Result))
            text_file.write("\n=====================================================")
            text_file.write("\nKGCA user input Config:\n")
            text_file.write(Dict2String(self.Config))
        
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
        KClusterMin = self.Config["KClusterMin"] - 1 # See #-----Kmeans
        KClusterMax = self.Config["KClusterMax"]
        KDistortions = self.KPopRes[CurrentGen]["KDistortions"]
        KExplainedVar = self.KPopRes[CurrentGen]["KExplainedVar"]
        KLeastImproveRate = self.Config["KLeastImproveRate"]
        KExplainedVarThres = self.Config["KExplainedVarThres"]
        SelectedK = self.KPopRes[CurrentGen]["SelectedK"]
        Kby = self.KPopRes[CurrentGen]["Kby"]
        # Elbow plot
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot(range(KClusterMin, KClusterMin+len(KDistortions)), KDistortions, marker='o', markersize=5, c = "blue")
        ax2.plot(range(KClusterMin, KClusterMin+len(KDistortions)), KExplainedVar, marker='o', markersize=5, c = "orange")
        ax.set_title(self.__name__ + " (Gen {})".format(CurrentGen))
        if Kby == "ImproveRate":
            ax.scatter(SelectedK, KDistortions[-2], s=100, 
                    facecolors='none', edgecolors='r', label = "  Selected K \n(Rate = {})".format(KLeastImproveRate))
            ax.legend(loc = "center right")
        elif Kby == "ExplainedVar":
            ax2.scatter(SelectedK, KExplainedVar[-1], s=100, 
                    facecolors='none', edgecolors='r', label = "  Selected K \n(Thres = {})".format(KExplainedVarThres))
            ax2.legend(loc = "center right")
            ax2.axhline(y=KExplainedVarThres, ls = "--", lw = 0.5, c = "grey")
        if Kby == "KMax":
            ax.scatter(SelectedK, KDistortions[-1], s=100, 
                    facecolors='none', edgecolors='r', label = "  Selected K \n(Reach K Max)".format(KLeastImproveRate))
            ax.legend(loc = "center right")
        ax.set_xlabel("Number of clusters (Max K = {})".format(KClusterMax))
        ax.set_ylabel("Distortion (within cluster sum of squares")    
        ax2.set_ylabel("Explained Variance")
        ax2.set_ylim([0,1])
        
        plt.show()