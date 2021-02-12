#%%
# Diverse model calibrations (DMC) genetic algorithm (GA).
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# DMC algorithm is based on (Williams et al., 2020).
# We generalized the code and add a mutation method call mutation_middle.
# Also, beside DMCGA class, we create DMCGA_Convertor class to help user convert back and forth between 1D array used by DMCGA and original parameter dataframes.
# 2021/02/11

from .SystemConrol import loadConfig
from joblib import Parallel, delayed    # For parallelization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import logging
import time
logger = logging.getLogger("HydroCNHS.DMC") # Get logger 

r"""
If NumSP = 0   => it should work
check function 
force parallel in HydroCNHS to stop
timeout (function) 
"""

r"""
Inputs = {"ParName":[], 
          "ParBound":[],  # [upper, low] or [4, 6, 9] Even for category type, it has to be numbers!
          "ParType":[],   # real or category
          "ParWeight":[], # Should be an array.
          "WD":}   
          
Config = {"NumSP":1,
          "PopSize": 30,            # Must be even.
          "MaxGen": 100,
          "SamplingMethod": "MC",
          "Tolerance":0.2,
          "NumEllite": 1,           # Ellite number for each SP. At least 1.
          "MutProb": 0.3,           # Mutation probability.
          "DropRecord": True,       # Population record will be dropped. However, ALL simulated results will remain. 
          "ParalCores": 2/None,     # This will overwrite system config.
          "AutoSave": True,         # Automatically save a model snapshot after each generation.
          "Printlevel": 10,         # Print out level. e.g. Every ten generations.
          "Plot": True              # Plot loss with Printlevel frequency.
          }
"""

class DMCGA(object):
    def __init__(self, LossFunc, Inputs, Config, Formatter = None, ContinueFile = None, Name = None):
               
        # Populate class attributions.
        self.LossFunc = LossFunc            # Loss function LossFunc(pop, Formatter, SubWDInfo = None).
                                             #pop is a parameter vector. 
                                             #Formatter can be obtained from class DMCGA_Convertor.
                                             #SubWDInfo = (CaliWD, CurrentGen, sp, k).
                                             #Lower bound of return value has to be 0.
        self.Inputs = Inputs                # Input ductionary.
        self.Inputs["ParWeight"] = np.array(Inputs["ParWeight"])    # Make sure it is an array.
        self.Config = Config                # Configuration for DMCGA.
        self.Formatter = Formatter          # Formatter is to convert 1D pop back into list of dataframe dictionaries for HydroCNHS simulation. (see class DMCGA_Convertor) 
        self.SysConfig = loadConfig()        #Load system config => Config.yaml (Default parallelization setting)
        self.NumPar = len(Inputs["ParName"])
        
        # If continue is True, load auto-saved pickle.
        if ContinueFile is not None:
            # Load autoSave pickle file!
            with open(ContinueFile, "rb") as f:
                Snapshot = pickle.load(f)
            # Load back all the previous class attributions.
            for key in Snapshot:
                setattr(self, key, Snapshot[key])
            self.Continue = True            # If it is continue run, no initialization is needed in "run".
            pass
        
        #---------- Auto save section ----------
        if ContinueFile is None:
            self.Continue = False   # If it is continue run, no initialization is needed in "run".
            # Generate index lists for later for loop and readibility.
            self.SPList = ["SP0"] + ["SP"+str(i+1) for i in range(Config["NumSP"])]
            
            # Populate initial counter.
            self.CurrentGen = 0
            
            # Initialize variables storage.
            self.Pop = {}           # Population of parameter set. Pop[gen][sp][k,s]
            self.PopRes = {}        # Simulation results (Loss values and min distance, D). PopRes[gen][sp][Loss or D]
            self.SPCentroid = {}            # SP[gen]
            self.Best = {"Loss":{}, "Index":{}}
            for sp in self.SPList:
                # Open np.empty, so nextGen() will always assign Best to the correct spots according CurrentGen (for continue run).
                self.Best["Loss"][sp] = np.empty(self.Config["MaxGen"]+1)  # +1 since including gen 0.
                self.Best["Index"][sp] = np.empty(self.Config["MaxGen"]+1) # +1 since including gen 0.  
            
            # Calculate scales for parameter normalization.
            self.BoundScale = []
            for i, ty in enumerate(Inputs["ParType"]):
                if ty == "real":
                    self.BoundScale.append(Inputs["ParBound"][i][1] - Inputs["ParBound"][i][0])
                elif ty == "category":
                    self.BoundScale.append(np.max(Inputs["ParBound"][i]) - np.min(Inputs["ParBound"][i]))
            self.BoundScale = np.array(self.BoundScale)     # Store in array type. 
            
            # Create calibration folder
            if Name is None:
                self.__name__ = "Calibration"
            else:
                self.__name__ = Name
            self.CaliWD = os.path.join(Inputs["WD"], self.__name__)
            # Create CaliWD directory
            if os.path.isdir(self.CaliWD) is not True:
                os.mkdir(self.CaliWD)
            else:
                logger.warning("!!! Current calibration folder exists. Default to overwrite the folder!!!\n{}".format(self.CaliWD))
        #---------------------------------------

    def MCSample(self, pop, ParBound, ParType):
        """Generate samples using Monte Carlo method.

        Args:
            pop (Array): 2D array. [PopSize, NumPar]
            NumPar (int): Number of parameters.
            ParBound (list): List of bounds for each parameters.
            ParType (list): List of parameter types. ["real" or "category"]

        Returns:
            array: Populated pop array.
        """
        PopSize = pop.shape[0]      # pop = [PopSize, NumPar]
        NumPar = pop.shape[1]
        for i in range(NumPar):
            if ParType[i] == "real":
                pop[:,i] = np.random.uniform(ParBound[i][0], ParBound[i][1], size = PopSize)  
            elif ParType[i] == "category":
                pop[:,i] = np.random.choice(ParBound[i], size = PopSize)
        return pop 
    
    def initialize(self, SamplingMethod = "MC", InitialPop = None):
        PopSize = self.Config["PopSize"]
        NumPar = self.NumPar
        ParBound = self.Inputs["ParBound"]
        ParType =  self.Inputs["ParType"]

        # Initialize storage space for generation 0.
        self.Pop[0] = {}
        self.PopRes[0] = {}
        self.SPCentroid[0] = {}
        
        # Populate pop for each SP.
        if InitialPop is None:      # Initialize parameters according to selected sampling method.
            for sp in self.SPList:
                pop = np.zeros((PopSize, NumPar))  # Create 2D array population for single generation.
                if SamplingMethod == "MC":
                    self.Pop[0][sp] = self.MCSample(pop, ParBound, ParType)
        else:                           # Initialize parameters with user inputs.
            self.Pop[0] = InitialPop
        
        # Initialize storage space for each SP
        for sp in self.SPList:
            self.PopRes[0][sp] = {}
            self.SPCentroid[0][sp] = {}
        
    def nextGen(self):
        LossFunc = self.LossFunc
        Formatter = self.Formatter
        CurrentGen = self.CurrentGen
        PopSize = self.Config["PopSize"]
        SPList = self.SPList
        NumSP = self.Config["NumSP"]
        NumPar = self.NumPar
        NumEllite = self.Config["NumEllite"]
        
        # Load parallelization setting (from user or system config)
        ParalCores = self.Config.get("ParalCores")
        if ParalCores is None:      # If user didn't specify, then we will use system default cores.
            ParalCores = self.SysConfig["Parallelization"]["Cores_DMCGA"]
        ParalVerbose = self.SysConfig["Parallelization"]["verbose"]
        
        #---------- Evaluation (Min) ----------
        # Note: Since HydroCHNS is a stochastic model, we will re simulate the ellite parameter set!!
        # In future, we can have an option for this to further reduce computational efficiency.
        # Evalute objective funtion (Loss function)
        # LossFunc(pop, Formatter, SubWDInfo = None); SubWDInfo = (CaliWD, CurrentGen, sp, k)
        LossParel = Parallel(n_jobs = ParalCores, verbose = ParalVerbose) \
                           ( delayed(LossFunc)\
                             (self.Pop[CurrentGen][sp][k], Formatter, (self.CaliWD, CurrentGen, sp, k)) \
                             for sp in SPList for k in range(PopSize) )  # Still go through entire Pop including ellites.
        # Get results
        for i, sp in enumerate(SPList):    # To fit two level for loop in joblib assignment.
            self.PopRes[CurrentGen][sp]["Loss"] = np.array(LossParel[i*PopSize : (i+1)*PopSize])
        #--------------------------------------
        
        #---------- Feasibility ----------
        # We define 1: feasible and 0: infeasible
        # Get the best sol from SP0
        SP0BestIndex = np.argmin(self.PopRes[CurrentGen]["SP0"]["Loss"])
        SP0Best = self.PopRes[CurrentGen]["SP0"]["Loss"][SP0BestIndex]
        self.Best["Loss"]["SP0"][CurrentGen] = SP0Best
        self.Best["Index"]["SP0"][CurrentGen] = SP0BestIndex
        if SP0Best == 0:
            SP0Best = 0.0001 # To prevent 0, which results in no tolerance for other SP. 
        
        # Determine the feasibility of each solution.
        Tol = self.Config["Tolerance"]
        for sp in SPList:    
            Feasibility = np.zeros(PopSize)
            Feasibility[self.PopRes[CurrentGen][sp]["Loss"] <= SP0Best*Tol] = 1   # Only valid when lower bound is zero
            self.PopRes[CurrentGen][sp]["Feasibility"] = Feasibility.astype(int)
        #---------------------------------
        
        #---------- Diversity ----------
        # Calculate fitness-weighted centriod 
        self.weight_fitness = {}
        for sp in SPList[1:]:   # No SP0
            # Calculate fitness weight using Loss. Lowest loss => 1; Highest loss => 0.
            Loss_sp = self.PopRes[CurrentGen][sp]["Loss"]
            temp = Loss_sp.argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = (PopSize-1 - np.arange(PopSize))      # Reverse ranks.
            self.weight_fitness[sp] = ranks = ranks/(PopSize-1) # Turn into linear scaling.
            # Average over pop to calculate centroid parameter set.
            pop_sp = self.Pop[CurrentGen][sp]                   # [PopSize, NumPar]
            pop_sp_w = np.multiply(pop_sp, self.weight_fitness[sp].reshape(PopSize,1))  # Broadcast weights into pop_sp
            self.SPCentroid[CurrentGen][sp]["Centroid"] = pop_sp_w.mean(axis = 0)   # Take average over PopSize (axis = 0)
            self.SPCentroid[CurrentGen][sp]["NormalizedCentroid"] = np.divide(self.SPCentroid[CurrentGen][sp]["Centroid"].reshape(1, NumPar),\
                                                                      self.BoundScale.reshape(1, NumPar))
        # Calculate minimum distance Dmin over q != p (No SP0)
        for sp in SPList[1:]:            # No SP0
            Distance = np.zeros((PopSize, NumSP))
            pop_sp = self.Pop[CurrentGen][sp]                   # [PopSize, NumPar]
            Nor_pop_sp = np.divide(pop_sp, self.BoundScale.reshape(1, NumPar))
            for i, sp_q in enumerate(SPList[1:]):   # No SP0
                if sp == sp_q:
                    Distance[:,i] = 1000000     # Assign a large number so it will never be chose when finding min.
                    # Calculate self distance.
                    Nor_centroid_sp_q = self.SPCentroid[CurrentGen][sp_q]["NormalizedCentroid"]
                    d = np.subtract(Nor_pop_sp, Nor_centroid_sp_q.reshape(1,NumPar))
                    # Multipy weights to each parameter.
                    d = np.multiply(d, self.Inputs["ParWeight"].reshape(1,NumPar))
                    self.PopRes[CurrentGen][sp]["SelfD"] = np.linalg.norm(d, axis = 1)         # l2norm 
                else:
                    Nor_centroid_sp_q = self.SPCentroid[CurrentGen][sp_q]["NormalizedCentroid"]
                    d = np.subtract(Nor_pop_sp, Nor_centroid_sp_q.reshape(1,NumPar))
                    # Multipy weights to each parameter.
                    d = np.multiply(d, self.Inputs["ParWeight"].reshape(1,NumPar))
                    Distance[:,i] = np.linalg.norm(d, axis = 1)         # l2norm 
            Dmin = Distance.min(axis = 1)  # For each k in pop, pick the min distance over SP q!=p.
            self.PopRes[CurrentGen][sp]["Dmin"] = Dmin
        #-------------------------------
        
        #---------- Selection ----------
        # Select ellites for each SP
        ElliteIndex = {}
        for sp in SPList:
            Loss = self.PopRes[CurrentGen][sp]["Loss"]  # Array
            if sp == "SP0":     # Select based on loss only.
                # argpartition has better efficient in searching n smallest values' index. Linear time.
                ElliteIndex[sp] = np.argpartition(Loss, NumEllite)[:NumEllite].astype(int)
            else:               # Select the most distant feasible solution.
                Feasibility_sp = self.PopRes[CurrentGen][sp]["Feasibility"]
                if np.sum(Feasibility_sp) <= NumEllite:     # Not enough feasible solutions.
                    ElliteIndex[sp] = np.argpartition(Loss, NumEllite)[:NumEllite].astype(int)  # Based on loss only. All feasible sols will be selected.
                    # Further sort the ElliteIndex to find out Best.
                    SPBestIndex = ElliteIndex[sp][  np.argmin( Loss[ElliteIndex[sp]] )  ]
                    SPBest = Loss[SPBestIndex]
                else:
                    Dmin_sp = self.PopRes[CurrentGen][sp]["Dmin"]
                    Dmin_sp = Dmin_sp*Feasibility_sp    # All Dmin of infeasible sols will become 0. Therefore, no chance to be selected.
                    # Further sort the ElliteIndex to find out Best.
                    ElliteIndex[sp] = np.argpartition(Loss, -NumEllite)[-NumEllite:].astype(int)    # return n largest Dmin index.
                    SPBestIndex = ElliteIndex[sp][  np.argmax( Loss[ElliteIndex[sp]] )  ]
                    SPBest = Loss[SPBestIndex]
                self.Best["Loss"][sp][CurrentGen] = SPBest
                self.Best["Index"][sp][CurrentGen] = SPBestIndex
    
        # Select parents for each SP
        # Binary tournament => Create a population of parents that is the same size as the original population PopSize.
        ParentIndex = {}
        for sp in SPList:
            ParentIndex[sp] = np.zeros(PopSize)
            CompetitorPair = np.random.randint(low = 0, high = PopSize, size=(PopSize,2)) # Randomly PopSize pairs (2 individuals). 
            Feasibility_sp = self.PopRes[CurrentGen][sp]["Feasibility"]
            TotalFeasibility_sp = np.sum(Feasibility_sp)    # How many feasible sols in total.
            
            if sp == "SP0" or TotalFeasibility_sp < 0.5*PopSize:    # If is SP0 or majority of pop is infeasible solution.
                Loss = self.PopRes[CurrentGen][sp]["Loss"]
                for k in range(PopSize):        # Select based on loss only.
                    pair = CompetitorPair[k,:]
                    ParentIndex[sp][k] = pair[  np.argmin( [Loss[pair[0]],  Loss[pair[1]]] )  ] 
            else:   # ~SP0
                Dmin_sp = self.PopRes[CurrentGen][sp]["Dmin"]
                Loss = self.PopRes[CurrentGen][sp]["Loss"]
                for k in range(PopSize):
                    pair = CompetitorPair[k,:]
                    pair_feasibility = [ Feasibility_sp[pair[0]],  Feasibility_sp[pair[1]] ]
                    SumFeasibility = np.sum(pair_feasibility)
                    if SumFeasibility == 1:     # One feasible sol: Select the feasible sol
                        ParentIndex[sp][k] = pair[  pair_feasibility.index(1)  ]    # pair_feasibility is a list.
                    elif SumFeasibility == 2:   # Two feasible sols: Select larger Dmin if both are feasible.
                        ParentIndex[sp][k] = pair[  np.argmax( [Dmin_sp[pair[0]], Dmin_sp[pair[1]]] )  ]
                    elif SumFeasibility == 0:   # None feasible sol: Select lower loss
                        ParentIndex[sp][k] = pair[  np.argmin( [Loss[pair[0]],  Loss[pair[1]]] )  ] 
            ParentIndex[sp] = ParentIndex[sp].astype(int)
        #-------------------------------
        
        #---------- Evolve ----------
        MutProb = self.Config["MutProb"]
        
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
        
        def Mutation_middle(child, parent1, parent2):
            MutSample_MC = self.MCSample(np.zeros((1,NumPar)), self.Inputs["ParBound"], self.Inputs["ParType"])
            ratio = np.random.random(NumPar)
            interval = np.abs(parent1 - parent2)
            P1_less_P2 = parent1 < parent2
            P2_less_P1 = parent2 < parent1
            P1_eq_P2 = parent2 == parent1
            Category = self.Inputs["ParType"] == "category"
            child[P1_less_P2] = parent1[P1_less_P2] + ratio[P1_less_P2]*interval[P1_less_P2]
            child[P2_less_P1] = parent2[P2_less_P1] + ratio[P2_less_P1]*interval[P2_less_P1]
            child[P1_eq_P2] = MutSample_MC.flatten()[P1_eq_P2]  # Since MutSample_MC.shape = (1, NumPar).
            child[Category] = MutSample_MC.flatten()[Category]  # Since MutSample_MC.shape = (1, NumPar).
            return child
        
        self.Pop[CurrentGen+1] = {}
        for sp in SPList:
            self.Pop[CurrentGen+1][sp] = np.zeros((PopSize, NumPar))
            Pop_sp = self.Pop[CurrentGen][sp]
            ParentIndex_sp = ParentIndex[sp]
            # Uniform crossover
            for p in range(int(PopSize/2)):     # PopSize must be even.
                parent1 = Pop_sp[int(ParentIndex_sp[2*p])]      # Make sure index is integer.
                parent2 = Pop_sp[int(ParentIndex_sp[2*p+1])]    # Make sure index is integer.
                child1 = UniformCrossover(parent1, parent2)
                child2 = UniformCrossover(parent1, parent2)
                child1 = Mutation(child1)
                child2 = Mutation_middle(child2, parent1, parent2)
                self.Pop[CurrentGen+1][sp][2*p] = child1
                self.Pop[CurrentGen+1][sp][2*p+1] = child2
            # Replace first n pop with ellites. 
            for i, e in enumerate(ElliteIndex[sp]):
                self.Pop[CurrentGen+1][sp][i] = Pop_sp[e]
        #----------------------------
        
        #---------- Prepare For Next Gen ----------
        self.PopRes[CurrentGen+1] = {}
        self.SPCentroid[CurrentGen+1] = {}
        for sp in self.SPList:
            self.PopRes[CurrentGen+1][sp] = {}
            self.SPCentroid[CurrentGen+1][sp] = {}
        #------------------------------------------
        return None
    
    def dropRecord(self):
        """Drop historical populations. However, we still keep historical tracking for all results.
        """
        if self.Config["DropRecord"] and (self.CurrentGen-1) >= 0:
            try:    # To make sure self.CurrentGen-1 exsit. When program shoutdown (Continue = Ture), we might encounter self.CurrentGen-1 has been deleted.
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
        # Setting timer
        start_time = time.monotonic()
        self.elapsed_time = 0
        
        SamplingMethod = self.Config["SamplingMethod"]
        MaxGen = self.Config["MaxGen"]
        AutoSave = self.Config["AutoSave"]
        
        # If it is a continuous run from previous shockdown work, we don't need initialization.
        if self.Continue is not True:
            self.initialize(SamplingMethod, InitialPop)
        else:
            logger.info("Continue from Gen {}.".format(self.CurrentGen))
            
        # Run the loop until reach maximum generation. (Can add convergent termination critiria in the future.)
        while self.CurrentGen <= MaxGen:
            self.nextGen()
            self.dropRecord()
            # If Autosave is True, a model snapshot (pickle file) will be saved at CaliWD.
            if AutoSave:
                self.autoSave()
            # Print output    
            if self.CurrentGen%self.Config["Printlevel"] == 0:
                pg = int(self.CurrentGen/(MaxGen/10))
                ProgressBar = "#"*pg+"-"*(10-pg)
                logger.info("{:4d}/{:4d}   |{}|.".format(self.CurrentGen, MaxGen, ProgressBar))
                if self.Config["Plot"]:
                    self.plotProgress()
            # Next generation
            self.CurrentGen += 1 
                            
                
        # Delete Pop with gen index = (MaxGen+1 -1)
        del self.Pop[self.CurrentGen]   
        del self.PopRes[self.CurrentGen]  
        del self.SPCentroid[self.CurrentGen]  
        
        # Extract solutions
        self.Solutions = {}
        for sp in self.SPList:
            self.Solutions[sp] = self.Pop[self.CurrentGen-1][sp][ int(self.Best["Index"][sp][self.CurrentGen-1]) ]
        
        # Count duration.
        elapsed_time = time.monotonic() - start_time
        self.elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        logger.info("Done! [{}]".format(self.elapsed_time))
    
    def plotProgress(self):
        BestLoss = self.Best["Loss"]
        x = np.arange(0, self.CurrentGen+1)
        fig, ax = plt.subplots()
        for sp in self.SPList:
            loss = BestLoss[sp][:self.CurrentGen+1]
            if sp == "SP0":
                ax.plot(x, loss, label = sp, linewidth = 2, color = "black")
            else:
                ax.plot(x, loss, label = sp)
        ax.set_xlim([0, self.Config["MaxGen"]+1])
        ax.set_ylim([0, BestLoss["SP0"][0]*1.1])
        ax.set_xlabel("Generation")
        ax.set_ylabel("Loss")
        ax.legend()
        
#%%
class DMCGA_Convertor(object):
    """This DMCGA_Convertor helps user to convert multiple parameter dataframe (can obtain nan values) into an 1D array (automatically exclude nan values) that can be used for DMCGA calibration. And the Formatter created by DMCGA_Convertor can be used to convert 1D array back to list of dataframe. Besides, we provide option for defining fixed values that doesn't need to be calibrated. 
    Note: Dataframe index is parameter names.
    """
    def __init__(self):
        pass
        
    def genFormatter(self, DFList, FixedParList = None):
        """[Include in genDMCGAInputs] Generate Formatter for given list of dataframe objects.  

        Args:
            DFList (list): A list of dataframes. Dataframe index is parameter names.
            FixedParList (list, optional): A list contains a list of fixed parameter names for each dataframe. Defaults to None.
        """
        Formatter = {"NumDF": None,
                    "ShapeList": [],
                    "ColNameList": [],
                    "IndexNameList": [],
                    "Index": [0],
                    "NoneIndex": None,
                    "FixedParList": None,
                    "FixedParValueList": []}
        if FixedParList is not None:
                Formatter["FixedParList"] = FixedParList
        
        Formatter["NumDF"] = len(DFList)
        VarArray = []        
        for i, df in enumerate(DFList):
            Formatter["ShapeList"].append(df.shape)
            Formatter["ColNameList"].append(df.columns.values)
            Formatter["IndexNameList"].append(df.index.values)
            # Store fixed par and replace their values with None in df.
            if FixedParList is not None:
                if FixedParList[i] == []:
                    Formatter["FixedParValueList"].append(None)
                else:
                    Formatter["FixedParValueList"].append(df.loc[FixedParList[i], :])
                    df.loc[FixedParList[i], :] = None
            # Convert to 1d array
            VarArray = VarArray + list(df.to_numpy().flatten("C"))    # [Row1, Row2, .....
            # Add Index
            Formatter["Index"].append(len(VarArray))
            
        VarArray = np.array(VarArray)       # list to array
        Formatter["NoneIndex"] = list(np.argwhere(np.isnan(VarArray)).flatten()) # Find index for np.nan
        self.Formatter = Formatter
    
    def genDMCGAInputs(self, WD, DFList, ParTypeDict, ParBoundDict, ParWeightDict = None, FixedParList = None):
        """Generate Inputs dictionary required for DMCGA.

        Args:
            WD (path): Working directory defined in the model.yaml.
            DFList (list): A list of dataframes. Dataframe index is parameter names.
            ParTypeDict (dict): A dictionary with key = parameter name and value = paremeter type [real/category]
            ParBoundDict (dict): A dictionary with key = parameter name and value = [lower bound, upper bound] or [1, 2, 3 ...]
            ParWeightDict (dict, optional): A dictionary with key = parameter name and value = weight (from SA). Defaults to None, weight = 1.
            FixedParList (list, optional): A list contains a list of fixed parameter names (don't need calibration) for each dataframe. Defaults to None.
        """
        # Compute Formatter
        self.genFormatter(DFList, FixedParList)
        Formatter = self.Formatter
        NoneIndex = Formatter["NoneIndex"]
        ParName = []
        ParType = []
        ParBound = []
        ParWeight = []
        # Form a list of above infomation (1D)
        for i in range(len(DFList)):
            ColNameList_d = Formatter["ColNameList"][i]
            IndexNameList_d = Formatter["IndexNameList"][i]
            for par in IndexNameList_d:
                for c in ColNameList_d:
                    ParName.append(str(par)+"|"+str(c))
                    ParType.append(ParTypeDict[par])
                    ParBound.append(ParBoundDict[par])
                    if ParWeightDict is None:
                        ParWeight.append(1)
                    else:
                        ParWeight.append(ParWeightDict[par])
        # Remove None index from Formatter. This include fixed par and par with None value. 
        def delete_multiple_element(list_object, indices):
            indices = sorted(indices, reverse=True)
            for idx in indices:
                if idx < len(list_object):
                    list_object.pop(idx)
        delete_multiple_element(ParName, NoneIndex)
        delete_multiple_element(ParBound, NoneIndex)
        delete_multiple_element(ParType, NoneIndex)
        delete_multiple_element(ParWeight, NoneIndex)
        Inputs = {"WD": WD, "ParName": ParName, "ParBound": ParBound, "ParType": ParType, "ParWeight": ParWeight}
        self.Inputs = Inputs
    
    @staticmethod   # staticmethod doesn't depends on object. It can be used independently.
    def to1DArray(DFList, Formatter):
        """Convert a list of dataframe to a 1D array following Formatter setting.

        Args:
            DFList (list): A list of dataframes. Dataframe index is parameter names.
            Formatter (dict): Generated by genFormatter or genDMCGAInputs. It is stored in attributions of the DMCGA_Convertor object.

        Returns:
            Array: 1D array.
        """
        VarArray = []        
        for df in DFList:
            # Convert to 1d array
            VarArray = VarArray + list(df.to_numpy().flatten("C"))   
            
        def delete_multiple_element(list_object, indices):
            indices = sorted(indices, reverse=True)
            for idx in indices:
                if idx < len(list_object):
                    list_object.pop(idx)
        delete_multiple_element(VarArray, Formatter["NoneIndex"])
        return np.array(VarArray)
    
    @staticmethod   # staticmethod doesn't depends on object. It can be used independently.
    def toDFList(VarArray, Formatter):
        """Convert 1D array back to a list of original dataframe based on Formatter.

        Args:
            VarArray (array): 1D array.
            Formatter (dict): Generated by genFormatter or genDMCGAInputs. It is stored in attributions of the DMCGA_Convertor object.

        Returns:
            list: A list of dataframes. Dataframe index is parameter names.
        """
        NoneIndex = Formatter["NoneIndex"]
        Index = Formatter["Index"]
        # Insert np.nan to VarArray following NoneIndex
        for i in NoneIndex:
            VarArray = np.insert(VarArray,i,np.nan)
        # Form DFList
        DFList = []
        for i in range(Formatter["NumDF"]):
            # 1d array to dataframe 
            df = np.reshape(VarArray[Index[i]: Index[i+1]], Formatter["ShapeList"][i], "C")
            df = pd.DataFrame(df)
            df.index = Formatter["IndexNameList"][i]
            df.columns = Formatter["ColNameList"][i]
            # Add fixed values back
            if Formatter["FixedParList"] is not None:
                df.loc[Formatter["FixedParList"][i],:] = Formatter["FixedParValueList"][i]
            DFList.append(df)
        return DFList

#%% DMCGA_Convertor Example
r"""
from pprint import pprint
# Randomly create dfs
df1 = pd.DataFrame([[1,2,3],[4,5,6]])
df2 = pd.DataFrame([[1,None,3],[4,5,6]])
df3 = pd.DataFrame([[1,2,3],[4,None,6]])
DFList = [df1,df2,df3]
for i, df in enumerate(DFList):
    df.index = ["A"+str(i+1),"B"+str(i+1)]
    df.columns = ["a","b","c"]
# Define parameter properties.    
ParTypeDict = {"A1": "real", "B1": "category",
               "A2": "real", "B2": "category",
               "A3": "real", "B3": "category"}
ParBoundDict = {"A1": [1,10], "B1": [1,2,3,4,5],
                "A2": [1,10], "B2": [4,5],
                "A3": [1,10], "B3": [1,4,5]}
ParWeightDict = {"A1":0.5, "B1":0.8,
                 "A2": 0.5, "B2": 0.8,
                 "A3": 0.5, "B3": 0.8}
FixedParList = [["A1"],[],[]]

# Create Convertor object
Convertor = DMCGA_Convertor()
# Run the Convertor
Convertor.genDMCGAInputs("haha", DFList, ParTypeDict, ParBoundDict, ParWeightDict, FixedParList)
# Take out Inputs and Formatter
Inputs = Convertor.Inputs
Formatter = Convertor.Formatter
pprint(Inputs)
pprint(Formatter)
# Convert dfs to 1D array
VarArray = Convertor.to1DArray(DFList, Formatter)
print("\nDfs to 1D array, which contains no nan and fixed parameters.")
print(VarArray)
# Convert 1D array back to dfs
dflist = Convertor.toDFList(VarArray, Formatter)
print("\n1D array back to dfs.")
dflist  
"""
# %%
