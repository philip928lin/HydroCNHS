import pandas as pd
import numpy as np
import ast

class Convertor(object):
    """GA_Convertor helps user to convert multiple parameter dataframe (can obtain nan values) into an 1D array (parameters for calibration, automatically exclude nan values) that can be used for DMCGA calibration. And the Formatter created by GA_Convertor can be used to convert 1D array back to a list of original dataframe. Besides, we provide option for defining fixed parameters, which will not enter the calibration process (exclude from the 1D array).
    Note: Dataframe index is parameter names.
    """
    def __init__(self):
        pass
        
    def genFormatter(self, DFList, FixedParList = None):
        """[Already included in genCaliInputs()] Generate Formatter for given list of dataframe objects.  

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
        VarArray[VarArray<-90]
        Formatter["NoneIndex"] = list(np.argwhere(np.isnan(VarArray)).flatten())    # Find index for np.nan values.
        self.Formatter = Formatter
    
    def genCaliInputs(self, WD, DFList, ParTypeDFList, ParBoundDFList, ParWeightDFList = None, FixedParList = None):
        """Generate Inputs dictionary required for DMCGA.

        Args:
            WD (path): Working directory defined in the model.yaml.
            DFList (list): A list of dataframes. Dataframe index is parameter names.
            ParTypeDFList (dict): Similar to DFList but with values => paremeter type [real/categorical]
            ParBoundDFList (dict): Similar to DFList but with values => paremeter type [real/categorical] = [lower bound, upper bound] or [1, 2, 3 ...]
            ParWeightDFList (dict, optional): Similar to DFList but with values => paremeter type [real/categorical] weight (from SA). Defaults to None, weight = 1.
            FixedParList (list, optional): A list contains a list of fixed parameter names (don't need calibration) for each dataframe. Defaults to None.
        """
        # Parse value from str to list or others.
        def parse(i):
            try:
                val = ast.literal_eval(i)
                return val
            except:
                return i           
        
        # Compute Formatter
        self.genFormatter(ParBoundDFList, FixedParList)     # We use ParBoundDFList to determine None.
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
                
            # Assignment starts here.    
            for par in IndexNameList_d:
                for c in ColNameList_d:
                    ParName.append(str(par)+"|"+str(c))
                    ParType.append(ParTypeDFList[i].loc[par,c])
                    ParBound.append(parse(ParBoundDFList[i].loc[par,c]))
                    if ParWeightDFList is None:
                        ParWeight.append(1)
                    else:
                        ParWeight.append(parse(ParWeightDFList[i].loc[par,c]))
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
            Formatter (dict): Generated by genFormatter or genDMCGAInputs. It is stored in attributions of the GA_Convertor object.

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
            Formatter (dict): Generated by genFormatter or genDMCGAInputs. It is stored in attributions of the GA_Convertor object.

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