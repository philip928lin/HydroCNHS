# Utility functions' module
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com)
# Last update at 2022/1/16.

import os
import ast
import itertools
import logging
from logging import config
import traceback
import numpy as np
import pandas as pd
import yaml
import ruamel.yaml  # For round trip modification (keep comments)
from copy import deepcopy
import importlib.util  # For importing customized module.

logger = logging.getLogger("HydroCNHS.sc")


# -----------------------------------------------
# ---------- Read and Write Functions ----------
# Acquire "this" file path.
this_dir, this_filename = os.path.split(__file__)


def set_logging_config(log_filename=None, log_level="WARNING"):
    """Set up logging config.

    Parameters
    ----------
    log_filename : str, optional
        Log filename, by default None
    """
    with open(os.path.join(this_dir, "Config_logging.yaml"), "rt") as file:
        logging_config = yaml.safe_load(file.read())
    if log_filename is not None:
        logging_config["loggers"]["HydroCNHS"]["handlers"] = ["console", "file"]
        if log_filename is True:
            logging_config["handlers"]["file"]["filename"] = "HydroCNHS.log"
        else:
            if ".log" not in log_filename:
                log_filename = log_filename + ".log"
            logging_config["handlers"]["file"]["filename"] = log_filename
    else:
        # Delete file handler.
        logging_config["handlers"].pop("file")

    # Set log level
    logging_config["loggers"]["HydroCNHS"]["level"] = log_level
    

    config.dictConfig(logging_config)


def load_model(model, checked=False, parsed=False, print_summary=True):
    """Load model.yaml or model dictionary.

    Parameters
    ----------
    model : str/dict
        Filename (e.g., model.yaml) or model dictionary.
    checked : bool, optional
        If True, no pre-check will be implemented, by default False.
    parsed : bool, optional
        If True, model will not be re-parses, by default False.
    print_summary : bool, optional
        Print the summary of loaded model, by default True.

    Returns
    -------
    dict
        Model dictionary.
    """
    logger = logging.getLogger("HydroCNHS")
    if isinstance(model, dict):
        model = model
    else:
        with open(model, "rt") as file:
            try:
                model = yaml.safe_load(file.read())
            except Exception as e:
                # Logs the error appropriately.
                logger.error(traceback.format_exc())
                return None

    # Check model is consist and correct.
    if checked is not True:
        if check_model(model) is not True:
            return None

    # Parse model
    if parsed is not True:
        model = parse_model(model, print_summary)

    return model


def write_model(model_dict, filename, org_model=None):
    """Write model dictionary to .yaml file

    If org_model is given, comments in the original file will be kept in the
    output model file.

    Parameters
    ----------
    model_dict : dict
        Model dictionary.
    filename : str
        Model configuration filename (e.g. model.yaml).
    org_model : str, optional
        Original model name (e.g. org_model.yaml), by default None.

    Note
    ----------
    Common error :
        1. value type is numpy.float64 => convert it to float.
    """
    if org_model is not None:  # Contain comments in the original model file.
        # Defaults to round-trip if no parameters given
        yaml_round = ruamel.yaml.YAML()
        with open(os.path.join(org_model), "rt") as file:
            model = yaml_round.load(file.read())
            model = model_dict
        with open(os.path.join(filename), "w") as file:
            yaml_round.dump(model, file)
    else:  # Dump without comments in the original model file.
        with open(filename, "w") as file:
            saved_model = yaml.safe_dump(
                model_dict, file, sort_keys=False, default_flow_style=None
            )
    logger.info("Model is saved at {}.".format(filename))


def write_model_to_df(model_dict, key_option=["Pars"], prefix=""):
    """Write model dictionary to dataframes given key options.

    This function will convert model dictionary to dataframes according to
    user's specified sections (key_option).

    Parameters
    ----------
    model_dict : dict
        Model dictionary.
    key_option : list, optional
        Output sections (e.g., Pars, Inputs, Attributes), by default ["Pars"].
    prefix : str, optional
        Prefix for the filenames, by default "".
    """

    def convert_dict_to_df(dict, colname):
        df = {}
        for k in dict:
            if isinstance(dict[k], (int, float, type(None))):
                df[k] = dict[k]
            elif isinstance(dict[k], list):
                for i, v in enumerate(dict[k]):
                    df[k + ".{}".format(i)] = v
            else:
                df[k] = str(dict[k])
        df = pd.DataFrame.from_dict(df, orient="index", columns=[colname])
        return df

    def merge_dicts(dict_list):
        dict_list = list(filter(None, dict_list))  # Remove None
        Len = len(dict_list)
        if Len == 0:
            return {}
        res_dict = deepcopy(dict_list[0])  # Avoid to modify the original dict.
        if Len > 1:
            for d in range(1, Len):
                res_dict.update(dict_list[d])
        return res_dict

    allowed_output_sections = ["RainfallRunoff", "Routing", "ABM", "Sediment"]
    section_list = [i for i in allowed_output_sections if i in model_dict]
    df_name = []
    output_df_list = []
    for s in section_list:
        if s == "RainfallRunoff" and key_option != ["Attributes"]:
            df_name.append(
                prefix + "Runoff_" + model_dict["WaterSystem"]["RainfallRunoff"]
            )
            df = pd.DataFrame()
            for sub in model_dict[s]:
                dict_list = [model_dict[s][sub].get(i) for i in key_option]
                merged_dict = merge_dicts(dict_list)
                converted_dict = convert_dict_to_df(merged_dict, sub)
                df = pd.concat([df, converted_dict], axis=1)
            output_df_list.append(df)
        elif s == "Routing" and key_option != ["Attributes"]:
            df_name.append(prefix + "Routing_" + model_dict["WaterSystem"]["Routing"])
            df = pd.DataFrame()
            for ro in model_dict[s]:
                for o in model_dict[s][ro]:
                    dict_list = [model_dict[s][ro][o].get(i) for i in key_option]
                    merged_dict = merge_dicts(dict_list)
                    converted_dict = convert_dict_to_df(merged_dict, (o, ro))
                    df = pd.concat([df, converted_dict], axis=1)
            output_df_list.append(df)
        elif s == "ABM":
            df_name.append(prefix + "ABM")
            df = pd.DataFrame()
            abm_setting = model_dict["WaterSystem"]["ABM"]
            AgTypes = (
                abm_setting["DamAPI"]
                + abm_setting["RiverDivAPI"]
                + abm_setting["InsituAPI"]
                + abm_setting["ConveyingAPI"]
            )
            for agtype in AgTypes:
                for ag in model_dict[s][agtype]:
                    dict_list = [model_dict[s][agtype][ag].get(i) for i in key_option]
                    merged_dict = merge_dicts(dict_list)
                    converted_dict = convert_dict_to_df(merged_dict, (ag, agtype))
                    df = pd.concat([df, converted_dict], axis=1)
            output_df_list.append(df)
        elif s == "Sediment":
            df_name.append(prefix + "Sediment")
            df = pd.DataFrame()
            for sub in model_dict[s]:
                dict_list = [model_dict[s][sub].get(i) for i in key_option]
                merged_dict = merge_dicts(dict_list)
                converted_dict = convert_dict_to_df(merged_dict, sub)
                df = pd.concat([df, converted_dict], axis=1)
            output_df_list.append(df)
    return output_df_list, df_name


def gen_default_bounds(model_dict, key_option=["Pars"]):
    """Generate default parameter bounds in the format of a list of
    DataFrames.

    Parameters
    ----------
    model_dict : dict
        Model dictionary.
    key_option : list, optional
        Output sections (e.g., Pars, Inputs, Attributes), by default ["Pars"].

    Returns
    -------
    tuple
        A list of parameter bound DataFrames, A list of DataFrame names.
    """
    df_list, df_name = write_model_to_df(model_dict, key_option, prefix="")

    # Default parameter bounds.
    gwlf_bounds = {
        "CN2": [25, 100],
        "IS": [0, 0.5],
        "Res": [0.001, 0.5],
        "Sep": [0, 0.5],
        "Alpha": [0, 1],
        "Beta": [0, 1],
        "Ur": [1, 15],
        "Df": [0, 1],
        "Kc": [0.5, 1.5],
    }
    abcd_bounds = {"a": [0, 1], "b": [0, 400], "c": [0, 1], "d": [0, 1], "Df": [0, 1]}
    routing_bounds = {
        "GShape": [1, 100],
        "GScale": [0.01, 150],
        "Velo": [0.5, 55],
        "Diff": [200, 4000],
    }
    sediment_bounds = {"CP": [1e-7, 1e-1], "Sa": [0, 4], "Sb": [1, 5], "Sq": [1, 5]}
    for df, name in zip(df_list, df_name):
        if "Runoff_GWLF" in name:
            for k, v in gwlf_bounds.items():
                df.loc[k, :] = str(v)
        elif "Runoff_ABCD" in name:
            for k, v in abcd_bounds.items():
                df.loc[k, :] = str(v)
        elif "Routing_Lohmann" in name:
            for k, v in routing_bounds.items():
                df.loc[k, df.loc[k, :].isna() == False] = str(v)
        elif "Sediment" in name:
            df.loc[:, :] = np.nan
            for k, v in sediment_bounds.items():
                df.loc[k, :] = str(v)
        else:
            for k in list(df.index):
                df.loc[k, df.loc[k, :].isna() == False] = str([0, 1])
    return df_list, df_name


def write_model_to_csv(folder_path, model_dict, key_option=["Pars"], prefix=""):
    """Write model (dictionary) to csv files.

    Parameters
    ----------
    folder_path : str
        Output folder directory.
    model_dict : dict
        Model dictionary.
    key_option : list, optional
        Output sections (e.g., Pars, Inputs, Attributes), by default ["Pars"].
    prefix : str, optional
        Prefix for the filenames, by default "".

    Returns
    -------
    list
        A list of filenames.
    """
    output_df_list, df_name = write_model_to_df(model_dict, key_option, prefix)
    df_name = [i + ".csv" for i in df_name]
    for i, df in enumerate(output_df_list):
        df.to_csv(os.path.join(folder_path, df_name[i]))
    logger.info("Output files {} at {}.".format(df_name, folder_path))
    return df_name  # CSV filenames


def load_df_to_model_dict(model_dict, df, section, key):
    """Load dataframe to model dictionary.

    Note that the column and index names have to be identical as outputted from
    write_model_to_df() function.

    Parameters
    ----------
    model_dict : dict
        Model dictionary.
    df : DataFrame
        Dataframe.
    section : str
        RainfallRunoff or Routing or ABM.
    key : str
        nputs or Pars or Attributes.

    Returns
    -------
    dict
        Updated model_dict.
    """

    def parse_df_to_dict(df):
        def parse(i):
            try:
                val = ast.literal_eval(i)
                return val
            except:
                return i

        def to_native_py_type(val):
            # Make sure the value is Native Python Type, which can be safely
            # dump to yaml.
            if "numpy" in str(type(val)):
                # Convert df's null value, which is np.nan into None. So yaml
                # will displat null instead of .nan
                if np.isnan(val):
                    return None
                else:
                    val = val.item()
                    if val == -99:
                        val = int(val)  # HydroCNHS special setting.
                    return val
            else:  # Sometime np.nan is not consider as numpy.float type.
                # Convert df's null value, which is np.nan into None. So yaml
                # will displat null instead of .nan
                if np.isnan(val):
                    return None
                else:
                    return val  # return other type

        # Form a corresponding dictionary for Pars.
        Col = [parse(i) for i in df.columns]  # Since we might have tuples.
        df.columns = Col
        # Identify list items by ".".
        Ind = [i.split(".")[0] for i in df.index]
        Ind_dup = {
            v: (Ind.count(v) if "." in df.index[i] else 0) for i, v in enumerate(Ind)
        }
        # Parse entire df
        df = df.applymap(parse)
        # Form structured dictionary with only native python type object.
        Dict = {}
        # Add each sub model setting to the dict. The value is either list or
        # str or float or int.
        for i in Col:
            temp = {}
            for par in Ind_dup:
                if Ind_dup[par] > 0:
                    temp[par] = list(
                        df.loc[[par + "." + str(k) for k in range(Ind_dup[par])], [i]]
                        .to_numpy()
                        .flatten()
                    )
                    temp[par] = [to_native_py_type(val) for val in temp[par]]
                    # Ensure W, Theta, LR ... lists won't contain None.
                    temp[par] = [val for val in temp[par] if val is not None]
                else:
                    temp[par] = to_native_py_type(df.loc[par, [i]].values[0])
            Dict[i] = temp
        return Dict

    df_dict = parse_df_to_dict(df)  # df to Dict

    # Replace the original modelDict accordingly.
    if section == "Runoff":
        runoff = model_dict["RainfallRunoff"]
        for sub in df_dict:
            runoff[sub][key] = df_dict[sub]
    elif section == "Routing":
        Routing = model_dict["Routing"]
        for roo in df_dict:
            Routing[roo[1]][roo[0]][key] = df_dict[roo]
    elif section == "ABM":
        ABM = model_dict["ABM"]
        for agagType in df_dict:
            ABM[agagType[1]][agagType[0]][key] = df_dict[agagType]
    if section == "Sediment":
        sed = model_dict["Sediment"]
        for sub in df_dict:
            sed[sub][key] = df_dict[sub]

    return model_dict


def load_customized_module_to_class(Class, module_name, path):
    """Load classes and functions in a user defined module (.py) into a given
    Class.

    Parameters
    ----------
    Class : class
        A class to collect classes and functions in a given module.
    module_name : str
        filename.py or filename.
    path : str
        Directory of filename.py.
    """
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(path, module_name)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Add classes and functions to a class.
    namespace = vars(module)
    public = (name for name in namespace if name[:1] != "_")
    for name in getattr(module, "__all__", public):
        setattr(Class, name, namespace[name])
    # globals().update(module.__dict__)   # Load all classes to globel.


def list_callable_public_object(obj):
    name_list = [
        name for name in dir(obj) if (callable(getattr(obj, name)) and name[:1] != "_")
    ]
    return name_list


# -----------------------------------------------

# -----------------------------------------
# ---------- Auxiliary Functions ----------


def dict_to_string(dictionary, indentor="  ", level=1):
    """Ture a dictionary into a printable string.

    Parameters
    ----------
    dictionary : dict
        A dictionary.
    indentor : str, optional
        Indentor, by default "  "

    Returns
    -------
    str
        String.
    """

    def dict_to_string_list(dictionary, indentor="  ", count=0, string=[]):
        for key, value in dictionary.items():
            string.append(indentor * count + str(key))
            if isinstance(value, dict) and count <= level:
                string = dict_to_string_list(value, indentor, count + 1, string)
            else:
                string.append(indentor * (count + 1) + str(value))
        return string

    return "\n".join(dict_to_string_list(dictionary, indentor))


def create_rn_gen(seed):
    """Create random number generator based on numpy module.

    Parameters
    ----------
    seed : int
        Random seed.

    Returns
    -------
    object
        Random number generator.
    """
    rn_gen = np.random.default_rng(seed)
    return rn_gen


# -----------------------------------------


# -----------------------------------------------
# ---------- Check and Parse Functions ----------
def check_model(model_dict):
    """Check the model dictionary

    Parameters
    ----------
    model_dict : dict
        Model dictionary

    Returns
    -------
    bool
        True if pass the check.
    """
    Pass = True

    # Need to make sure simulation period is longer than a month (GWLF part)
    # Name of keys (Agent and subbasin name) cannot be dulicated.
    # Name of keys (Agent and subbasin name) cannot have "."

    Pass = check_WS(model_dict)
    Pass = check_RainfallRunoff(model_dict)

    if model_dict.get("Routing") is not None and model_dict.get("ABM") is not None:
        Pass = check_agent_in_routing(model_dict)
    return Pass


def parse_model(model_dict, print_summary=True):
    """Parse model dictionary. Populate SystemParsedData.

    Args:
        model_dict (dict): Load from model.yaml.

    Returns:
        dict: model_dict
    """
    model_dict["SystemParsedData"] = {}

    if model_dict.get("Routing") is not None:
        model_dict = parse_sim_seq(model_dict, print_summary)
    return model_dict


def check_WS(model_dict):
    Pass = True
    ws = model_dict["WaterSystem"]

    # --- Check keys
    # "StartDate", "EndDate", "DataLength" are checked below
    ideal_keys = ["NumSubbasins", "Outlets", "RainfallRunoff", "Routing", "ABM"]
    ws_items = list(ws.keys())
    if all(item in ws_items for item in ideal_keys) is False:
        logger.error(
            "Missing items in WaterSystem setting. {}".format(
                set(ideal_keys) - set(ws_items)
            )
        )
        Pass = False

    data_length = ws.get("DataLength")
    end_date = ws.get("EndDate")
    if data_length is not None:
        end_date = pd.to_datetime(ws["StartDate"], format="%Y/%m/%d") + pd.DateOffset(
            data_length - 1
        )
        ws["EndDate"] = end_date.strftime("%Y/%m/%d")
    elif end_date is not None:
        end_date = pd.to_datetime(ws["EndDate"], format="%Y/%m/%d")
        start_date = pd.to_datetime(ws["StartDate"], format="%Y/%m/%d")
        ws["DataLength"] = (end_date - start_date).days + 1
    else:
        logger.error(
            "Either DataLength or EndDate has to be provided in "
            + "WaterSystem setting."
        )
        Pass = False

    outlets = ws["Outlets"]
    if len(outlets) != ws["NumSubbasins"]:
        logger.error("Outlets is inconsist to NumSubbasins.")
        Pass = False

    if len(outlets) != len(set(outlets)):
        logger.error("Duplicates exist in Outlets.")
        Pass = False

    if any("." in o for o in outlets):
        logger.error('"." is not allowed in the outlet name.')
        Pass = False

    # --- Check selected RainfallRunoff model.
    runoff_model = ws["RainfallRunoff"]
    runoff_options = ["GWLF", "ABCD", "Other"]
    if runoff_model not in runoff_options:
        logger.error(
            "Invalid rainfall-runoff model {}. Acceptable options: {}".format(
                runoff_model, runoff_options
            )
        )
        Pass = False
    return Pass


def check_RainfallRunoff(model_dict):
    runoff = model_dict["RainfallRunoff"]
    Pass = True
    # --- Check keys
    sb_list = model_dict["WaterSystem"]["Outlets"]
    ideal_keys = set(sb_list)
    runoff_keys = set(runoff)
    if runoff_keys != ideal_keys:
        logger.error(
            "Inconsist RainfallRunoff keys {}\nto {}".format(runoff_keys, ideal_keys)
        )
        Pass = False

    # --- Check subbasins
    ideal_keys2 = set(["Inputs", "Pars"])
    for sb in sb_list:
        sb_keys = set(runoff[sb])
        if sb_keys != ideal_keys2:
            logger.error(
                "Inconsist RainfallRunoff keys {}\nto {}".format(sb_keys, ideal_keys)
            )
            Pass = False
    return Pass


def check_agent_in_routing(model_dict):
    # To make sure InStreamAgentInflows outlets are assigned in the routing
    # section.
    # Missing InsituAgent.
    Pass = True
    routing = model_dict["Routing"]
    abm = model_dict["ABM"]
    ws = model_dict["WaterSystem"]
    # --- Check In-stream agents' Links and collect instream_Ag_inflows:
    # DamAPI is eligble in the routing setting.
    # DamAPI' agents will completely split the water system into half.
    instream_ag_inflows = []
    instream_ag_types = ws["ABM"]["DamAPI"]
    for ISagType in instream_ag_types:
        for ag in abm[ISagType]:
            links = abm[ISagType][ag]["Inputs"]["Links"]
            inflow_outlets = [
                outlet for outlet in links if links[outlet] == -1 and outlet != ag
            ]
            if inflow_outlets == []:
                logger.error(
                    "[Check model failed] No inflow outlets "
                    + "(e.g., Links:{InflowOutlet: -1}) are found "
                    + "for {}.".format(ag)
                )
                Pass = False
            if links.get(ag) is None:
                logger.info("Auto-fill outflow link for {}.".format(ag))
                abm[ISagType][ag]["Inputs"]["Links"][ag] = 1
            instream_ag_inflows += inflow_outlets
    # Eliminate duplicates.
    instream_ag_inflows = list(set(instream_ag_inflows))

    # --- Check InStreamAgents' inflow outlets are in RoutingOutlets.
    routing_outlets = list(routing.keys())
    for vro in instream_ag_inflows:
        if vro not in routing_outlets:
            logger.error(
                "[Check model failed] Cannot find in-stream agent's "
                + "inflow outlets in the Routing section. Routing "
                + "outlets should include {}.".format(vro)
            )
            Pass = False
        else:  # Check if start belong to others routing_outlets' starts.
            for start in routing[vro]:
                for ro in routing_outlets:
                    if ro != vro:
                        if any(start in others for others in routing[ro]):
                            # If it is in the upstream of VirROutlet, it's
                            # fine.
                            if ro in routing[vro]:
                                pass
                            else:
                                logging.error(
                                    "[Check model failed] "
                                    + "{} sub-basin outlet ".format(start)
                                    + "shouldn't be in {}'s ".format(ro)
                                    + "(routing outlet) catchment outlets due "
                                    + "to the seperation of in-stream control "
                                    + "agent {}.".format(vro)
                                )
                                Pass = False

    # --- RiverDivAPI will not add a new routing outlet (agent itself)
    # like DamAPI do to split the system into half but their diverting
    # outlet must be in routing outlets.
    river_div_ag_inflows = []
    for river_div_type in ws["ABM"]["RiverDivAPI"]:
        for ag in abm[river_div_type]:
            links = abm[river_div_type][ag]["Inputs"]["Links"]

            def get_div(x):
                try:
                    return x <= 0
                except:
                    return x[-1] == "Minus"

            divert_outlet = [
                outlet for outlet in links if get_div(links[outlet]) and outlet != ag
            ]
            if divert_outlet == []:
                logger.error(
                    "[Check model failed] No diverted outlets "
                    + "(e.g., Links:{DivertOutlet: -1}) are found "
                    + "for {}.".format(ag)
                )
                Pass = False
            river_div_ag_inflows += divert_outlet
    # Eliminate duplicates.
    river_div_ag_inflows = list(set(river_div_ag_inflows))

    # --- Check RiverDivAgents' diverting outlets are in routing_outlets.
    for vro in river_div_ag_inflows:
        if vro not in routing_outlets:
            logger.error(
                "[Check model failed] Cannot find RiverDivAgent "
                + "diverting outlets in the Routing section. Routing "
                + "outlets should include {}.".format(vro)
            )
            Pass = False

    return Pass


def form_sim_seq(node_list, back_tracking_dict):
    key_set = set(back_tracking_dict)
    if isinstance(node_list, str):
        if node_list not in key_set:
            return [node_list]  # If there is only a single routing outlet.
        node_list = [node_list]
    else:
        for node in node_list:
            if node not in key_set:
                back_tracking_dict[node] = []
    sim_seq = node_list

    def find_upstream_node(node):
        upstream_node = set(back_tracking_dict[node])
        not_top_node = upstream_node.intersection(key_set)
        top_node = upstream_node - not_top_node
        return list(not_top_node), list(top_node)

    while node_list != []:
        layer_nodes = []
        for node in node_list:
            not_top_node, top_node = find_upstream_node(node)
            sim_seq = top_node + sim_seq
            layer_nodes = layer_nodes + not_top_node
        # Back filling
        sim_seq = layer_nodes + sim_seq
        node_list = layer_nodes
    return sim_seq


def update_sim_seq_with_group(sim_seq, group, back_tracking_dict):
    branch_dict = {}
    for node in group:
        branch_dict[node] = form_sim_seq(node, back_tracking_dict)[:-1]
    update_seq = list(itertools.chain.from_iterable(branch_dict.values()))
    update_seq = update_seq + group
    if len(update_seq) != len(set(update_seq)):
        print(
            "Given group {} is not eligible. Update simulation sequence fail.".format(
                group
            )
        )
        return sim_seq  # Not update
    else:
        remain_node = [n for n in sim_seq if n not in update_seq]
        update_seq = update_seq + remain_node
        return update_seq


def parse_sim_seq(model_dict, print_summary=True):
    model_dict["SystemParsedData"]["SimSeq"] = None
    model_dict["SystemParsedData"]["AgSimSeq"] = None
    model_dict["SystemParsedData"]["RoutingOutlets"] = None
    model_dict["SystemParsedData"]["DamAgents"] = None
    model_dict["SystemParsedData"]["RiverDivAgents"] = None
    model_dict["SystemParsedData"]["InsituAgents"] = None
    model_dict["SystemParsedData"]["ConveyingAgents"] = None
    model_dict["SystemParsedData"]["BackTrackingDict"] = None
    model_dict["SystemParsedData"]["Edges"] = None
    # Store info for constructing routing UH for conveyed water of those node.
    model_dict["SystemParsedData"]["ConveyToNodes"] = []

    # ----- Collect in-stream agents
    ## In-stream agents here means those agents will re-define the streamflow
    # completely. Namely, only DamAgents.
    ## RiverDivAPI, in our definition, only modify the streamflow.

    if (
        model_dict.get("ABM") is not None
        and model_dict["WaterSystem"]["ABM"] is not None
    ):
        ws_abm = model_dict["WaterSystem"]["ABM"]
        abm = model_dict["ABM"]
        for ag_types in ["DamAPI", "RiverDivAPI", "InsituAPI", "ConveyingAPI"]:
            agents = []
            for ag_type in ws_abm[ag_types]:
                for end in abm[ag_type]:
                    # "RiverDivAPI" since we can multiple ag divert from
                    # the same gauge. We modify streamflow directly.
                    agents.append(end)
            model_dict["SystemParsedData"][ag_types[:-3] + "Agents"] = agents
    instream_agents = model_dict["SystemParsedData"]["DamAgents"]
    if instream_agents is None:
        instream_agents = []

    # ----- Step1: Form the simulation sequence of routing outlets and in-stream
    # control agents -----
    # Collected edges and track-back dictionary.
    # Edges can be further used for plotting routing simulation schema using
    # networkx.
    edges = []
    back_tracking_dict = {}
    routing = model_dict["Routing"]
    routing_outlets = list(routing.keys())
    for end in routing_outlets:
        for start in routing[end]:
            if start == end:
                pass  # We don't add self edge.
            else:
                # Eliminate only-sub-basin outlets. If need full stream node
                # sequence, remove this.
                if start in routing_outlets + instream_agents:
                    edges.append((start, end))
                    if back_tracking_dict.get(end) is None:
                        back_tracking_dict[end] = [start]
                    else:
                        back_tracking_dict[end].append(start)

    # Add in-stream agents connectinons if ABM sections exists.
    if (
        model_dict.get("ABM") is not None
        and model_dict["WaterSystem"]["ABM"] is not None
    ):
        ws_abm = model_dict["WaterSystem"]["ABM"]
        instream_ag_types = ws_abm["DamAPI"]
        for ag_type in instream_ag_types:
            for end in abm[ag_type]:
                links = abm[ag_type][end]["Inputs"]["Links"]
                # Since InStreamAgents will completely redefine the streamflow,
                # the inflow factor is defined to be -1.
                # Note that we will not use this factor anywhere in the
                # simulation.
                # end != node condition is designed for DamDivAgentTypes, where
                # its DM is diversion that has factor equal to -1 or < 0 as
                # well.
                # We don't want those edges (e.g. (end, end)) to ruin the
                # simulation sequence formation.
                inflow_nodes = [
                    node for node in links if links[node] == -1 and end != node
                ]
                for start in inflow_nodes:
                    edges.append((start, end))
                    if back_tracking_dict.get(end) is None:
                        back_tracking_dict[end] = [start]
                    else:
                        back_tracking_dict[end].append(start)
    model_dict["SystemParsedData"]["BackTrackingDict"] = back_tracking_dict
    model_dict["SystemParsedData"]["Edges"] = edges

    # Construct sim_seq
    if back_tracking_dict == {}:  # If there is only a single outlet!
        sim_seq = routing_outlets
    else:
        # end nodes - start nodes = watershed outlets node_list.
        node_list = list(set(back_tracking_dict.keys()) - set([i[0] for i in edges]))
        sim_seq = form_sim_seq(node_list, back_tracking_dict)
        group_nodes = model_dict["WaterSystem"].get("NodeGroups")
        if group_nodes is not None:
            for group in group_nodes:
                sim_seq = update_sim_seq_with_group(sim_seq, group, back_tracking_dict)
    model_dict["SystemParsedData"]["SimSeq"] = sim_seq
    # Sort RoutingOutlets based on SimSeq
    model_dict["SystemParsedData"]["RoutingOutlets"] = [
        ro for ro in sim_seq if ro in routing_outlets
    ]
    # --------------------------------------------------------------------------

    # ----- Step2: Form AgSim dictionary -----
    # Aggregate to only SimSeq's node.
    if (
        model_dict.get("ABM") is not None
        and model_dict["WaterSystem"]["ABM"] is not None
    ):
        ws_abm = model_dict["WaterSystem"]["ABM"]
        dam_ag_types = ws_abm["DamAPI"]
        river_ag_types = ws_abm["RiverDivAPI"]
        insitu_ag_types = ws_abm["InsituAPI"]
        convey_ag_types = ws_abm["ConveyingAPI"]
        # Ordered routing_outlets
        routing_outlets = model_dict["SystemParsedData"]["RoutingOutlets"]

        # Institutions: Institutions = {"AgType":{"Name": []}}
        institutions = ws_abm.get("Institutions")
        if institutions is None:
            institutions = []

        def search_routing_outlet(agQ):
            """Find in which routing outlet first need agQ to adjust original
            Q. (from upstream).
            Args:
                agQ (str): Outlets connection of an agent.
            Returns:
                str: Routing outlet.
            """
            if agQ in routing_outlets:
                return agQ
            for ro in routing_outlets:
                if agQ in routing[ro]:
                    return ro

        def create_empty_list(dict, key):
            if dict.get(key) is None:
                dict[key] = []

        ag_sim_seq = {}
        priority = {}
        ag_sim_seq["AgSimMinus"] = {}
        priority["AgSimMinus"] = {}
        ag_sim_seq["AgSimPlus"] = {}
        priority["AgSimPlus"] = {}
        for ss in sim_seq:
            ag_sim_seq["AgSimMinus"][ss] = {}
            priority["AgSimMinus"][ss] = {}
            ag_sim_seq["AgSimPlus"][ss] = {}
            priority["AgSimPlus"][ss] = {}

        for ag_type in dam_ag_types:
            if ag_type in institutions:
                ag_list = institutions[ag_type].keys()  # Use the group name
            else:
                ag_list = abm[ag_type].keys()
            # Note that in-stream agent (ag) will be in SimSeq
            for ag in ag_list:
                create_empty_list(ag_sim_seq["AgSimPlus"][ag], "DamAgents")
                create_empty_list(priority["AgSimPlus"][ag], "DamAgents")
                # No AgSimMinus is needed since in-stream agents replace
                # original streamflow.
                ag_sim_seq["AgSimPlus"][ag]["DamAgents"].append((ag, ag))
                # In-stream agent always has priority 0.
                priority["AgSimPlus"][ag]["DamAgents"].append(0)

        for ag_type in river_ag_types:
            if ag_type in institutions:
                ag_list = institutions[ag_type].keys()  # Use the group name
                group = True
            else:
                ag_list = abm[ag_type].keys()
                group = False
            for ag in ag_list:
                if group:
                    # Use the setting of the first member in the group
                    member = institutions[ag_type][ag][0]
                else:
                    member = ag
                links = abm[ag_type][member]["Inputs"]["Links"]
                # "list" is our special offer to calibrate return flow factor
                # (Inputs).
                plus = []
                minus = []
                for p in links:
                    if isinstance(links[p], list):
                        if links[p][-1] == "Minus":
                            minus.append(p)
                        else:
                            plus.append(p)
                    else:
                        if links[p] >= 0:
                            plus.append(p)
                        else:
                            minus.append(p)

                # plus = [p if isinstance(links[p], list) else p
                # if links[p] >= 0 else None for p in links]
                # minus = [None if isinstance(links[p], list) else p
                # if links[p] <= 0 else None for p in links]
                # plus = list(filter(None, plus))        # Drop None in a list.
                # minus = list(filter(None, minus))      # Drop None in a list.
                for p in plus:
                    # Return flow can be added to non-routing outlets. So we
                    # need to find the associate routing outlet in the SimSeq.
                    ro = search_routing_outlet(p)
                    create_empty_list(ag_sim_seq["AgSimPlus"][ro], "RiverDivAgents")
                    create_empty_list(priority["AgSimPlus"][ro], "RiverDivAgents")
                    ag_sim_seq["AgSimPlus"][ro]["RiverDivAgents"].append((ag, p))
                    priority["AgSimPlus"][ro]["RiverDivAgents"].append(
                        abm[ag_type][member]["Inputs"]["Priority"]
                    )
                for m in minus:
                    # RiverDivAgents diverted outlets should belong to one
                    # routing outlet.
                    # ro = search_routing_outlet(m)
                    ro = m
                    create_empty_list(ag_sim_seq["AgSimMinus"][ro], "RiverDivAgents")
                    create_empty_list(priority["AgSimMinus"][ro], "RiverDivAgents")
                    ag_sim_seq["AgSimMinus"][ro]["RiverDivAgents"].append((ag, m))
                    priority["AgSimMinus"][ro]["RiverDivAgents"].append(
                        abm[ag_type][member]["Inputs"]["Priority"]
                    )

        for ag_type in convey_ag_types:
            if ag_type in institutions:
                ag_list = institutions[ag_type].keys()  # Use the group name
                group = True
            else:
                ag_list = abm[ag_type].keys()
                group = False
            for ag in ag_list:
                if group:
                    # Use the setting of the first member in the group
                    member = institutions[ag_type][ag][0]
                else:
                    member = ag
                links = abm[ag_type][member]["Inputs"]["Links"]
                plus = []
                minus = []
                for p in links:
                    if links[p] >= 0:
                        plus.append(p)
                    else:
                        minus.append(p)

                model_dict["SystemParsedData"]["ConveyToNodes"] = plus
                for p in plus:
                    # Flow can be added to non-routing outlets. So we
                    # need to find the associate routing outlet in the SimSeq.
                    ro = search_routing_outlet(p)
                    create_empty_list(ag_sim_seq["AgSimPlus"][ro], "ConveyingAgents")
                    create_empty_list(priority["AgSimPlus"][ro], "ConveyingAgents")
                    ag_sim_seq["AgSimPlus"][ro]["ConveyingAgents"].append((ag, p))
                    priority["AgSimPlus"][ro]["ConveyingAgents"].append(
                        abm[ag_type][member]["Inputs"]["Priority"]
                    )
                for m in minus:
                    # ConveyingAgents convey from a routing outlet.
                    # ro = search_routing_outlet(m)
                    ro = m
                    create_empty_list(ag_sim_seq["AgSimMinus"][ro], "ConveyingAgents")
                    create_empty_list(priority["AgSimMinus"][ro], "ConveyingAgents")
                    ag_sim_seq["AgSimMinus"][ro]["ConveyingAgents"].append((ag, m))
                    priority["AgSimMinus"][ro]["ConveyingAgents"].append(
                        abm[ag_type][member]["Inputs"]["Priority"]
                    )

        for ag_type in insitu_ag_types:
            # insitu_ag_types is a simple diversion agent type, which only
            # divert/add water from a single sub-basin.
            # Runoff of the max(sub-basin - InsituDiv, 0)
            # Note that it divert from runoff of a single sub-basin not river
            # and no return flow option.
            if ag_type in institutions:
                ag_list = institutions[ag_type].keys()  # Use the group name
                group = True
            else:
                ag_list = abm[ag_type].keys()
                group = False
            for ag in ag_list:
                if group:
                    # Use the setting of the first member in the group
                    member = institutions[ag_type][ag][0]
                else:
                    member = ag
                links = abm[ag_type][member]["Inputs"]["Links"]
                # No special "list" offer to calibrate return flow factor
                # (Inputs).
                # No return flow option.
                plus = []
                minus = []
                for p in links:
                    if links[p] >= 0:
                        plus.append(p)
                    else:
                        minus.append(p)
                for p in plus:
                    # Return flow can be added to non-routing outlets. So we
                    # need to find the associate routing outlet in the SimSeq.
                    ro = search_routing_outlet(p)
                    create_empty_list(ag_sim_seq["AgSimPlus"][ro], "InsituAgents")
                    create_empty_list(priority["AgSimPlus"][ro], "InsituAgents")
                    ag_sim_seq["AgSimPlus"][ro]["InsituAgents"].append((ag, p))
                    priority["AgSimPlus"][ro]["InsituAgents"].append(
                        abm[ag_type][member]["Inputs"]["Priority"]
                    )
                for m in minus:
                    ro = search_routing_outlet(m)
                    create_empty_list(ag_sim_seq["AgSimMinus"][ro], "InsituAgents")
                    create_empty_list(priority["AgSimMinus"][ro], "InsituAgents")
                    ag_sim_seq["AgSimMinus"][ro]["InsituAgents"].append((ag, m))
                    priority["AgSimMinus"][ro]["InsituAgents"].append(
                        abm[ag_type][member]["Inputs"]["Priority"]
                    )

        # Sort agents based on their piorities
        for pm in ag_sim_seq:
            for ag_types in ag_sim_seq[pm]:
                for ro in ag_sim_seq[pm][ag_types]:
                    agents = ag_sim_seq[pm][ag_types][ro]
                    piorities = priority[pm][ag_types][ro]
                    agents = [ag for _, ag in sorted(zip(piorities, agents))]
                    # Remove duplicated ags.
                    ag_sim_seq[pm][ag_types][ro] = list(set(agents))

        model_dict["SystemParsedData"]["AgSimSeq"] = ag_sim_seq
        # ----------------------------------------
    if print_summary:
        summary_dict = {}
        for i in [
            "SimSeq",
            "RoutingOutlets",
            "DamAgents",
            "ConveyingAgents",
            "RiverDivAgents",
            "InsituAgents",
            "AgSimSeq",
        ]:
            summary_dict[i] = model_dict["SystemParsedData"][i]
        parsed_model_summary = dict_to_string(summary_dict, indentor="  ")
        logger.info("Parsed model data summary:\n" + parsed_model_summary)
    return model_dict


# -------------------------------------
