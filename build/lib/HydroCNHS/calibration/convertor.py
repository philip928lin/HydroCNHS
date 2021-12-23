# Convertor module.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# Last update at 2021/12/22.
import ast
import pandas as pd
import numpy as np

class Convertor(object):
    def __init__(self):
        """Convertor helps user to convert multiple parameter dataframes (can
        contain nan values) into an 1D array (parameters for calibration,
        automatically exclude nan values) that can be used for GA calibration.
        And the formatter created by Convertor can be used to convert 1D array
        back to a list of original dataframes. Besides, we provide option for
        defining fixed parameters, which those parameters will not enter the
        calibration process (exclude from the 1D array).
        Note: Dataframe index is parameter names.
        """
        pass
        
    def gen_formatter(self, df_list, fixed_par_list=None):
        """Generate formatter for given list of dataframe objects. This is 
        already included in gen_cali_inputs().

        Parameters
        ----------
        df_list : list
            A list of parameter dataframes. Dataframe index is parameter names.
        fixed_par_list : list, optional
            A list contains a list of tuples of fixed parameter location like
            [(["CN2"], ["outlet1", "outlet2"]), (["b"], ["outlet1"])], by
            default None. The number of tuples in the list is corresponding to
            the number of dataframes.
            
        Returns
        -------
        dict
            formattor.
        """
        
        for i in range(len(df_list)):
            # Convert index and column into String, since tuple is not directly
            # callable.
            parsed_index = [str(item) for item in df_list[i].index]
            parsed_col = [str(item) for item in df_list[i].columns]
            df_list[i].index = parsed_index
            df_list[i].columns = parsed_col
            
        formatter = {"num_of_df": None,
                    "shape_list": [],
                    "col_name_list": [],
                    "index_name_list": [],
                    "index": [0],
                    "none_index": None,
                    "fixed_par_list": None,
                    "fixed_par_val_list": []}
        if fixed_par_list is not None:
                formatter["fixed_par_list"] = fixed_par_list
        
        formatter["num_of_df"] = len(df_list)
        var_array = []        
        for i, df in enumerate(df_list):
            formatter["shape_list"].append(df.shape)
            formatter["col_name_list"].append(df.columns.values)
            formatter["index_name_list"].append(df.index.values)
            # Store fixed par and replace their values with None in df.
            
            if fixed_par_list is not None:
                fixed_par_val_list = []
                for tup in fixed_par_list[i]:
                    if tup[1] is ":":       # For all columns.
                        tup = (tup[0], list(df))
                    if tup[0] is ":":       # For all index.
                        tup = (list(df.index), tup[1])
                    value = df.loc[tup[0], tup[1]].to_numpy()
                    fixed_par_val_list.append(value)
                    df.loc[tup[0], tup[1]] = None
                formatter["fixed_par_val_list"].append(fixed_par_val_list)
            # Convert to 1d array
            # Ensure when None occur in df, it will be replace by np.nan.
            df = df.replace({None: np.nan})
            # [Row1, Row2, .....
            var_array = var_array + list(df.to_numpy().flatten("C"))
            # Add index (where it ends in the 1D array)
            formatter["index"].append(len(var_array))
            
        var_array = np.array(var_array) # list to array
        formatter["none_index"] = list(
            np.argwhere(np.isnan(var_array)).flatten()
            )   # Find index for np.nan values.
        self.formatter = formatter
        return formatter
    
    
    def gen_cali_inputs(self, wd, df_list, par_bound_df_list,
                        fixed_par_list=None):
        """Generate inputs dictionary required for calibration.

        Parameters
        ----------
        wd : str
            Working directory.
        df_list : list
            A list of parameter dataframes. Dataframe index is parameter names.
        par_bound_df_list : list
            A list of parameter bound dataframes.
        fixed_par_list : list, optional
            A list contains a list of tuples of fixed parameter location like
            [(["CN2"], ["outlet1", "outlet2"]), (["b"], ["outlet1"])], by
            default None. The number of tuples in the list is corresponding to
            the number of dataframes.
            
        Returns
        -------
        dict
            Input for calibration.
        """
        # Parse value from str to list or others.
        def parse(i):
            try:
                val = ast.literal_eval(i)
                return val
            except:
                return i           
        
        # Compute formatter
        # We use par_bound_df_list to determine None.
        self.gen_formatter(df_list, fixed_par_list)
        formatter = self.formatter
        none_index = formatter["none_index"]
        par_name = []
        par_bound = []
        # Form a list of above infomation (1D)
        for i in range(len(df_list)):
            col_name_list_d = formatter["col_name_list"][i]
            index_name_list_d = formatter["index_name_list"][i]
            # Make sure index and column is callable and identical to df_list.
            par_bound_df_list[i].index = index_name_list_d
            par_bound_df_list[i].columns = col_name_list_d
                
            # Assignment starts here.    
            for par in index_name_list_d:
                for c in col_name_list_d:
                    par_name.append(str(par)+"|"+str(c))
                    par_bound.append(parse(par_bound_df_list[i].loc[par,c]))
        # Remove elements in None index from formatter. This includes fixed
        # pars and pars with None values. 
        def delete_multiple_element(list_object, indices):
            indices = sorted(indices, reverse=True)
            for idx in indices:
                if idx < len(list_object):
                    list_object.pop(idx)
        delete_multiple_element(par_name, none_index)
        delete_multiple_element(par_bound, none_index)
        inputs = {"wd": wd, "par_name": par_name, "par_bound": par_bound}
        self.inputs = inputs
        return inputs
    
    @staticmethod
    def to_1D_array(df_list, formatter):
        """Convert a list of dataframes to a 1D array.

        Parameters
        ----------
        df_list : list
            A list of parameter dataframes. Dataframe index is parameter names.
        formatter : dict
            Formatter, generated by gen_formatter or gen_cali_inputs. It is
            stored in attributes of the Convertor object.

        Returns
        -------
        array
            1D array.
        """
        var_array = []        
        for df in df_list:
            # Convert to 1d array
            var_array = var_array + list(df.to_numpy().flatten("C"))   
            
        def delete_multiple_element(list_object, indices):
            indices = sorted(indices, reverse=True)
            for idx in indices:
                if idx < len(list_object):
                    list_object.pop(idx)
        delete_multiple_element(var_array, formatter["none_index"])
        
        return np.array(var_array)
    
    @staticmethod
    def to_df_list(var_array, formatter):
        """Convert 1D array back to a list of dataframes.

        Parameters
        ----------
        var_array : array
            1D array.
        formatter : dict
            Formatter, generated by gen_formatter or gen_cali_inputs. It is
            stored in attributes of the Convertor object.

        Returns
        -------
        list
            A list of dataframes.
        """
        none_index = formatter["none_index"]
        index = formatter["index"]
        # Insert np.nan to var_array following none_index
        for i in none_index:
            var_array = np.insert(var_array,i,np.nan)
            
        # Form df_list
        df_list = []
        for i in range(formatter["num_of_df"]):
            # 1d array to dataframe 
            df = np.reshape(var_array[index[i]: index[i+1]],
                            formatter["shape_list"][i], "C")
            df = pd.DataFrame(df)
            df.index = formatter["index_name_list"][i]
            df.columns = formatter["col_name_list"][i]
            # Add fixed values back
            if formatter["fixed_par_list"] is not None:
                for ii, tup in enumerate(formatter["fixed_par_list"][i]):
                    if tup[1] is ":":       # For all columns.
                        tup = (tup[0], list(df))
                    if tup[0] is ":":       # For all index.
                        tup = (list(df.index), tup[1])
                    aa = formatter["fixed_par_val_list"][i][ii]
                    df.loc[tup[0], tup[1]] = aa
            df_list.append(df)
            
        return df_list
    
    
    # Archive
    r"""
    def gen_cali_inputs(self, wd, df_list, par_bound_df_list,
                        par_type_df_list=["real"], par_weight_df_list=None,
                        fixed_par_list=None):
        #""Generate inputs dictionary required for calibration.

        Parameters
        ----------
        wd : str
            Working directory.
        df_list : list
            A list of parameter dataframes. Dataframe index is parameter names.
        par_bound_df_list : list
            A list of parameter bound dataframes.
        par_type_df_list : list, optional
            Paremeter type, by default ["real"].
        par_weight_df_list : [type], optional
            A list of parameter weight dataframes, by default None
        fixed_par_list : list, optional
            A list contains a list of tuples of fixed parameter location like
            [(["CN2"], ["outlet1", "outlet2"]), (["b"], ["outlet1"])], by
            default None. The number of tuples in the list is corresponding to
            the number of dataframes.
            
        Returns
        -------
        dict
            Input for calibration.
        #""
        # Parse value from str to list or others.
        def parse(i):
            try:
                val = ast.literal_eval(i)
                return val
            except:
                return i           
        
        # Compute formatter
        # We use par_bound_df_list to determine None.
        self.gen_formatter(df_list, fixed_par_list)
        formatter = self.formatter
        none_index = formatter["none_index"]
        par_name = []
        par_type = []
        par_bound = []
        par_weight = []
        # Form a list of above infomation (1D)
        for i in range(len(df_list)):
            col_name_list_d = formatter["col_name_list"][i]
            index_name_list_d = formatter["index_name_list"][i]
            # Make sure index and column is callable and identical to df_list.
            if (isinstance(par_type_df_list[i], str) is False
                and par_type_df_list[i] != "real"):
                par_type_df_list[i].index = index_name_list_d
                par_type_df_list[i].columns = col_name_list_d
            par_bound_df_list[i].index = index_name_list_d
            par_bound_df_list[i].columns = col_name_list_d
            if par_weight_df_list is not None:
                par_weight_df_list[i].index = index_name_list_d
                par_weight_df_list[i].columns = col_name_list_d
                
            # Assignment starts here.    
            for par in index_name_list_d:
                for c in col_name_list_d:
                    par_name.append(str(par)+"|"+str(c))
                    if (isinstance(par_type_df_list[i], str) is False
                        and par_type_df_list[i] != "real"):
                        par_type.append(par_type_df_list[i].loc[par,c])
                    else:
                        par_type.append(par_type_df_list[i])
                    par_bound.append(parse(par_bound_df_list[i].loc[par,c]))
                    if par_weight_df_list is None:
                        par_weight.append(1)
                    else:
                        par_weight.append(parse(par_weight_df_list[i].loc[par,c]))
        # Remove elements in None index from formatter. This includes fixed
        # pars and pars with None values. 
        def delete_multiple_element(list_object, indices):
            indices = sorted(indices, reverse=True)
            for idx in indices:
                if idx < len(list_object):
                    list_object.pop(idx)
        delete_multiple_element(par_name, none_index)
        delete_multiple_element(par_bound, none_index)
        delete_multiple_element(par_type, none_index)
        delete_multiple_element(par_weight, none_index)
        inputs = {"wd": wd, "par_name": par_name, "par_bound": par_bound,
                  "par_type": par_type, "par_weight": par_weight}
        self.inputs = inputs
        return inputs
    """