#%%
# Diverse model calibrations (DMC) genetic algorithm (GA).
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# DMC algorithm is based on (Williams et al., 2020) https://doi.org/10.1016/j.envsoft.2020.104831.
# We generalized the code and add a mutation method call mutation_middle.
# However, we deactivate mutation_middle for DMC. This function helps convergence but restricts exploration in DMC case.
# Also, beside DMCGA class, we create DMCGA_Convertor class to help user convert back and forth between 1D array used by DMCGA and original parameter dataframes.
# 2021/02/11

from .SystemConrol import loadConfig, Dict2String   # HydroCNHS module
from joblib import Parallel, delayed                # For parallelization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import pickle
import os
import logging
import time
logger = logging.getLogger("HydroCNHS.DMC") # Get logger 

r"""
Need to be added sometime.
Check function 
Force parallel in HydroCNHS to stop
Timeout (function) 
"""

r"""
Inputs = {"ParName":[], 
          "ParBound":[],  # [upper, low] or [4, 6, 9] Even for categorical type, it has to be numbers!
          "ParType":[],   # real or categorical
          "ParWeight":[], # Should be an array.
          "WD":}   
          
Config = {"NumSP":1,                # Number of sub-populations.
          "PopSize": 30,            # Population size. Must be even.
          "MaxGen": 100,            # Maximum generation.
          "SamplingMethod": "LHC",  # MC: Monte Carlo sampling method. LHC: Latin Hyper Cube. (for initial pop)
          "Tolerance":1.2,          # >= 1 
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
    def __init__(self, LossFunc, Inputs, Config, Formatter = None, ContinueFile = None, Name = "Calibration"):
        """Diverse model calibrations (DMC) genetic algorithm (GA) object.

        Args:
            LossFunc (function): Loss function => LossFunc(pop, Formatter, SubWDInfo = None) and return loss, which has lower bound 0.
            Inputs (dict): Inputs dictionary, which can be generated by DMCGA_Convertor. It contains ParName, ParBound, ParType, ParWeight, and WD.
            Config (dict): Config dictionary, which contains NumSP, PopSize, MaxGen, SamplingMethod, Tolerance, NumEllite, MutProb, DropRecord, ParalCores (optional), AutoSave, Printlevel, and Plot.
            Formatter (dict, optional): Formatter dictionary created by DMCGA_Convertor. This will be further feed back to LossFunc for user to convert 1D array back to original format to run HydroCNHS. Defaults to None.
            ContinueFile (str, optional): AutoSave.pickle directory to continue the previous run. Defaults to None.
            Name (str, optional): Name of the DMCGA object, corresponding to the created sub-folder name. Defaults to "Calibration".
        """
               
        # Populate class attributions.
        self.LossFunc = LossFunc                # Loss function LossFunc(pop, Formatter, SubWDInfo = None) return loss, which has lower bound 0.
                                                    #pop is a parameter vector. 
                                                    #Formatter can be obtained from class DMCGA_Convertor.
                                                    #SubWDInfo = (CaliWD, CurrentGen, sp, k).
                                                    #Lower bound of return value has to be 0.
        self.Inputs = Inputs                    # Input ductionary.
        self.Inputs["ParWeight"] = np.array(Inputs["ParWeight"])    # Make sure it is an array.
        self.Config = Config                    # Configuration for DMCGA.
        self.Formatter = Formatter              # Formatter is to convert 1D pop back into list of dataframe dictionaries for HydroCNHS simulation. (see class DMCGA_Convertor) 
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
            self.SPList = ["SP0"] + ["SP"+str(i+1) for i in range(Config["NumSP"])]
            self.CurrentGen = 0     # Populate initial counter for generation.
            
            # Initialize variables storage.
            self.Pop = {}           # Population of parameter set. Pop[gen][sp]: [k,s] (2D array); k is index of members, and s is index of parameters.
            self.PopRes = {}        # Simulation results of each members. PopRes[gen][sp][Loss/Dmin/Feasibility/SelfD]: 1D array with length of population size.
            self.SPCentroid = {}    # Centroid of each SP. SPCentroid[gen][sp][Centroid/NormalizedCentroid]: 1D array with length of number of calibrated parameters.
            self.Best = {"Loss":{}, # Best loss value and index of corresponding member in Pop[gen][sp]. Best[Loss/Index][sp]: 1D array with length of MaxGen.
                         "Index":{}}
            for sp in self.SPList:
                # Open np.empty, so nextGen() will always assign Best to the correct spots according CurrentGen (for continue run).
                self.Best["Loss"][sp] = np.empty(self.Config["MaxGen"]+1)  # +1 since including gen 0.
                self.Best["Index"][sp] = np.empty(self.Config["MaxGen"]+1) # +1 since including gen 0.  
            
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
            SamplingMethod (str, optional): Selected method for generate initial population members. Currently, only MC can be assigned. Defaults to "MC" => Monte Carlo.
            InitialPop (dict, optional): User-provided initial Pop[0]. InitialPop[sp]: [PopSize, NumPar] (2D array). Defaults to None.
        """
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
                elif SamplingMethod == "LHC":
                    self.Pop[0][sp] = self.LatinHyperCubeSample(pop, ParBound, ParType)
        else:                       # Initialize parameters with user inputs.
            self.Pop[0] = InitialPop
        
        # Initialize storage space for each SP
        for sp in self.SPList:
            self.PopRes[0][sp] = {}
            self.SPCentroid[0][sp] = {}
        
    def nextGen(self):
        """Complete all simulations and generate next generation of Pop.
        Detail procedure, please see Algorithm 1 in (Williams et al., 2020) https://doi.org/10.1016/j.envsoft.2020.104831.
        """
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
                             (self.Pop[CurrentGen][sp][k], Formatter, (self.CaliWD, CurrentGen, sp, k)) \
                             for sp in SPList for k in range(PopSize) )  # Still go through entire Pop including ellites.
        # Get results
        for i, sp in enumerate(SPList):    # To fit two level for loop in joblib assignment.
            self.PopRes[CurrentGen][sp]["Loss"] = np.array(LossParel[i*PopSize : (i+1)*PopSize])
        #--------------------------------------
        
        #---------- Feasibility ----------
        # We define 1: feasible sol and 0: infeasible sol.
        # Get the best sol of SP0, which serves as the reference for feasibility.
        SP0BestIndex = np.argmin(self.PopRes[CurrentGen]["SP0"]["Loss"])
        SP0Best = self.PopRes[CurrentGen]["SP0"]["Loss"][SP0BestIndex]
        self.Best["Loss"]["SP0"][CurrentGen] = SP0Best
        self.Best["Index"]["SP0"][CurrentGen] = SP0BestIndex
        
        
        # Determine the feasibility of each solution in SP.
        if SP0Best == 0:
            SP0Best = 0.0001 # To prevent 0, which results in no tolerance for other SP. 
        Tol = self.Config["Tolerance"]      # Should be >= 1
        for sp in SPList:    
            Feasibility = np.zeros(PopSize)
            Feasibility[self.PopRes[CurrentGen][sp]["Loss"] <= SP0Best*Tol] = 1   # Only valid when lower bound is zero
            self.PopRes[CurrentGen][sp]["Feasibility"] = Feasibility.astype(int)
        #---------------------------------
        
        #---------- Diversity ----------
        # Calculate fitness-weighted centriod for each SP != SP0.
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
            self.SPCentroid[CurrentGen][sp]["Centroid"] = pop_sp_w.mean(axis = 0)       # Take average over PopSize (axis = 0)
            self.SPCentroid[CurrentGen][sp]["NormalizedCentroid"] = np.divide(self.SPCentroid[CurrentGen][sp]["Centroid"].reshape(1, NumPar),\
                                                                      self.BoundScale.reshape(1, NumPar))
        # Calculate minimum distance Dmin over q != p (No SP0)
        # Note: All parameters have benn normalized with their scales = paramter's range.
        for sp in SPList[1:]:            # No SP0
            Distance = np.zeros((PopSize, NumSP))
            pop_sp = self.Pop[CurrentGen][sp]       # [PopSize, NumPar]
            Nor_pop_sp = np.divide(pop_sp, self.BoundScale.reshape(1, NumPar))
            for i, sp_q in enumerate(SPList[1:]):   # No SP0
                if sp == sp_q:
                    Distance[:,i] = 1000000         # Assign a large number so it will never be chosen when finding min.
                    # Calculate self-distance (SelfD).
                    Nor_centroid_sp_q = self.SPCentroid[CurrentGen][sp_q]["NormalizedCentroid"]
                    d = np.subtract(Nor_pop_sp, Nor_centroid_sp_q.reshape(1,NumPar))
                    # Multipy weights to each parameter.
                    d = np.multiply(d, self.Inputs["ParWeight"].reshape(1,NumPar))
                    self.PopRes[CurrentGen][sp]["SelfD"] = np.linalg.norm(d, axis = 1)  # l2norm 
                else:
                    Nor_centroid_sp_q = self.SPCentroid[CurrentGen][sp_q]["NormalizedCentroid"]
                    d = np.subtract(Nor_pop_sp, Nor_centroid_sp_q.reshape(1,NumPar))
                    # Multipy weights to each parameter.
                    d = np.multiply(d, self.Inputs["ParWeight"].reshape(1,NumPar))
                    Distance[:,i] = np.linalg.norm(d, axis = 1)                         # l2norm 
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
                ElliteIndex[sp] = np.argpartition(Loss, NumEllite)[:NumEllite].astype(int)          # return n smallest Loss index.
            else:               # Select the most distant feasible solution.
                Feasibility_sp = self.PopRes[CurrentGen][sp]["Feasibility"]
                if np.sum(Feasibility_sp) <= NumEllite:     # Not enough feasible solutions.
                    # Based on loss only. All feasible sols will be automatically selected.
                    ElliteIndex[sp] = np.argpartition(Loss, NumEllite)[:NumEllite].astype(int)      # return n smallest Loss index.
                    # Further sort the ElliteIndex to find out Best.
                    SPBestIndex = ElliteIndex[sp][  np.argmin( Loss[ElliteIndex[sp]] )  ]
                    SPBest = Loss[SPBestIndex]
                else:
                    Dmin_sp = self.PopRes[CurrentGen][sp]["Dmin"]
                    Dmin_sp = Dmin_sp*Feasibility_sp    # All Dmin of infeasible sols will become 0. Therefore, no chance to be selected.
                    # Further sort the ElliteIndex to find out Best.
                    ElliteIndex[sp] = np.argpartition(Dmin_sp, -NumEllite)[-NumEllite:].astype(int)  # return n largest Dmin index.
                    SPBestIndex = ElliteIndex[sp][  np.argmax( Dmin_sp[ElliteIndex[sp]] )  ]
                    SPBest = Loss[SPBestIndex]
                self.Best["Loss"][sp][CurrentGen] = SPBest
                self.Best["Index"][sp][CurrentGen] = SPBestIndex
    
        # Select parents for each SP
        # Binary tournament => Create a population of parents that is the same size as the original population PopSize.
        # We store index k, instead of the whole parameter vector.
        ParentIndex = {}
        for sp in SPList:
            ParentIndex[sp] = np.zeros(PopSize)
            CompetitorPair = np.random.randint(low = 0, high = PopSize, size=(PopSize,2)) # Randomly PopSize pairs (2 individuals). 
            Feasibility_sp = self.PopRes[CurrentGen][sp]["Feasibility"]
            TotalFeasibility_sp = np.sum(Feasibility_sp)    # How many feasible sols in total.
            
            # If is SP0 or majority of pop is infeasible solution, we select parents based on loss only.
            if sp == "SP0" or TotalFeasibility_sp < 0.5*PopSize:    
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
            ParentIndex[sp] = ParentIndex[sp].astype(int)   # Make sure index is integer.
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
        
        # Mutation_middle is deactivate for now.
        def Mutation_middle(child, parent1, parent2): 
            MutSample_MC = self.MCSample(np.zeros((1,NumPar)), self.Inputs["ParBound"], self.Inputs["ParType"])
            ratio = np.random.random(NumPar)
            interval = np.abs(parent1 - parent2)
            P1_less_P2 = parent1 < parent2
            P2_less_P1 = parent2 < parent1
            P1_eq_P2 = parent2 == parent1
            Categorical = self.Inputs["ParType"] == "categorical"
            child[P1_less_P2] = parent1[P1_less_P2] + ratio[P1_less_P2]*interval[P1_less_P2]
            child[P2_less_P1] = parent2[P2_less_P1] + ratio[P2_less_P1]*interval[P2_less_P1]
            child[P1_eq_P2] = MutSample_MC.flatten()[P1_eq_P2]  # Since MutSample_MC.shape = (1, NumPar).
            child[Categorical] = MutSample_MC.flatten()[Categorical]  # Since MutSample_MC.shape = (1, NumPar).
            return child
        
        self.Pop[CurrentGen+1] = {}
        for sp in SPList:
            self.Pop[CurrentGen+1][sp] = np.zeros((PopSize, NumPar))
            Pop_sp = self.Pop[CurrentGen][sp]
            ParentIndex_sp = ParentIndex[sp]
            # Uniform crossover
            for p in range(int(PopSize/2)):     # PopSize must be even.
                parent1 = Pop_sp[ParentIndex_sp[2*p]]     
                parent2 = Pop_sp[ParentIndex_sp[2*p+1]]   
                child1 = UniformCrossover(parent1, parent2)
                child2 = UniformCrossover(parent1, parent2)
                child1 = Mutation(child1)
                child2 = Mutation(child2)       # Deactivate: Mutation_middle(child2, parent1, parent2)
                self.Pop[CurrentGen+1][sp][2*p] = child1
                self.Pop[CurrentGen+1][sp][2*p+1] = child2
            # Replace first n pop with ellites. 
            for i, e in enumerate(ElliteIndex[sp]):
                self.Pop[CurrentGen+1][sp][i] = Pop_sp[e]
        #----------------------------
        
        #---------- Prepare For Next Gen ----------
        # Open store space for next generation.
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
        del self.PopRes[self.CurrentGen]  
        del self.SPCentroid[self.CurrentGen]  
        
        #----- Extract solutions
        self.Solutions = {}
        for sp in self.SPList:
            self.Solutions[sp] = {}
            self.Solutions[sp]["Par"] = self.Pop[self.CurrentGen-1][sp][ int(self.Best["Index"][sp][self.CurrentGen-1]) ]
            self.Solutions[sp]["Loss"] = self.Best["Loss"][sp][self.CurrentGen-1]
        
        #----- Count duration.
        elapsed_time = time.monotonic() - start_time
        self.elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        logger.info("Done! [{}]".format(self.elapsed_time))
        logger.info("Solutions:\n" + Dict2String(self.Solutions))
    
    def plotProgress(self):
        """Plot Loss of all SPs. To visualize the convergence. 
        """
        BestLoss = self.Best["Loss"]
        x = np.arange(0, self.CurrentGen+1)
        fig, ax = plt.subplots()
        for sp in self.SPList:
            loss = BestLoss[sp][:self.CurrentGen+1]
            if sp == "SP0":
                ax.plot(x, loss, label = sp, linewidth = 2, color = "black")
            else:
                ax.plot(x, loss, label = sp)
        ax.set_title(self.__name__)
        ax.set_xlim([0, self.Config["MaxGen"]+1])
        ax.set_ylim([0, BestLoss["SP0"][0]*1.1])
        ax.set_xlabel("Generation")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.show()
        
#%%
class DMCGA_Convertor(object):
    """DMCGA_Convertor helps user to convert multiple parameter dataframe (can obtain nan values) into an 1D array (parameters for calibration, automatically exclude nan values) that can be used for DMCGA calibration. And the Formatter created by DMCGA_Convertor can be used to convert 1D array back to a list of original dataframe. Besides, we provide option for defining fixed parameters, which will not enter the calibration process (exclude from the 1D array).
    Note: Dataframe index is parameter names.
    """
    def __init__(self):
        pass
        
    def genFormatter(self, DFList, FixedParList = None):
        """[Already included in genDMCGAInputs()] Generate Formatter for given list of dataframe objects.  

        Args:
            DFList (list): A list of dataframes. Dataframe index is parameter names.
            FixedParList (list, optional): A list contains a list of tuples of fixed parameter loc [e.g. (["CN2"], ["S1", "S2"])] for each dataframe. Defaults to None.
        """
        for i in range(len(DFList)):
            # Convert index and column into String, since tuple is not directly callable.
            ParsedIndex = [str(item) for item in DFList[i].index]
            ParsedCol = [str(item) for item in DFList[i].columns]
            DFList[i].index = ParsedIndex
            DFList[i].columns = ParsedCol
            
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
                    for tup in FixedParList[i]:
                        Value = df.loc[tup[0], tup[1]].to_numpy()
                        Formatter["FixedParValueList"].append(Value)
                        df.loc[tup[0], tup[1]] = None
            # Convert to 1d array
            VarArray = VarArray + list(df.to_numpy().flatten("C"))    # [Row1, Row2, .....
            # Add Index (where it ends in the 1D array)
            Formatter["Index"].append(len(VarArray))
            
        VarArray = np.array(VarArray)       # list to array
        Formatter["NoneIndex"] = list(np.argwhere(np.isnan(VarArray)).flatten())    # Find index for np.nan values.
        self.Formatter = Formatter
    
    def genDMCGAInputs(self, WD, DFList, ParTypeDFList, ParBoundDFList, ParWeightDFList = None, FixedParList = None, Parse = False):
        """Generate Inputs dictionary required for DMCGA.

        Args:
            WD (path): Working directory defined in the model.yaml.
            DFList (list): A list of dataframes. Dataframe index is parameter names.
            ParTypeDict (dict): A dictionary with key = parameter name and value = paremeter type [real/categorical]
            ParBoundDict (dict): A dictionary with key = parameter name and value = [lower bound, upper bound] or [1, 2, 3 ...]
            ParWeightDict (dict, optional): A dictionary with key = parameter name and value = weight (from SA). Defaults to None, weight = 1.
            FixedParList (list, optional): A list contains a list of fixed parameter names (don't need calibration) for each dataframe. Defaults to None.
        """
        # Parse df to make sure the consistency of data type.
        def parse(Series):
            Series = list(Series)
            for i, v in enumerate(Series):
                try:
                    Series[i] = ast.literal_eval(v)
                except:
                    Series[i] = v
            return Series       
        
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
            # Make sure index and column is callable and identical to DFList.
            ParTypeDFList[i].index = IndexNameList_d
            ParTypeDFList[i].columns = ColNameList_d
            ParBoundDFList[i].index = IndexNameList_d
            ParBoundDFList[i].columns = ColNameList_d
            if ParWeightDFList is not None:
                ParWeightDFList[i].index = IndexNameList_d
                ParWeightDFList[i].columns = ColNameList_d
            if Parse:   # Parse string list or tuple to list or tuple. 
                ParBoundDFList[i].apply(parse, axis=0)
                
            # Assignment starts here.    
            for par in IndexNameList_d:
                for c in ColNameList_d:
                    ParName.append(str(par)+"|"+str(c))
                    ParType.append(ParTypeDFList[i].loc[par,c])
                    ParBound.append(ParBoundDFList[i].loc[par,c])
                    if ParWeightDFList is None:
                        ParWeight.append(1)
                    else:
                        ParWeight.append(ParWeightDFList[i].loc[par,c])
        # Remove elements in None index from Formatter. This includes fixed pars and pars with None values. 
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
                for ii, tup in enumerate(Formatter["FixedParList"][i]):
                    df.loc[tup[0], tup[1]] = Formatter["FixedParValueList"][i][ii]
            DFList.append(df)
        return DFList
