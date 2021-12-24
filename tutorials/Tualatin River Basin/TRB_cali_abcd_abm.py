import os
import pandas as pd
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt
import HydroCNHS
import HydroCNHS.calibration as cali

##### Path and Load Model Test
# Get this file directory.
prj_path, this_filename = os.path.split(__file__)
model_path = os.path.join(prj_path, "Template_for_calibration", "TRB_dm_abcd.yaml")
bound_path = os.path.join(prj_path, "ParBound")
wd = prj_path

# Update model paths.
model_dict = HydroCNHS.load_model(model_path)
model_dict["Path"]["WD"] = wd
model_dict["Path"]["Modules"] = os.path.join(prj_path, "ABM_modules")

##### Gen cali information
df_list, df_name = HydroCNHS.write_model_to_df(model_dict, key_option=["Pars"])
par_bound_df_list = [
    pd.read_csv(os.path.join(bound_path, "abcd_par_bound.csv"), index_col=[0]),
    pd.read_csv(os.path.join(bound_path, "routing_par_bound.csv"), index_col=[0]),
    pd.read_csv(os.path.join(bound_path, "abm_par_bound_dm.csv"), index_col=[0])]
converter = cali.Convertor()
converter.gen_cali_inputs(wd, df_list, par_bound_df_list)
formatter = converter.formatter
cali_inputs = converter.inputs

with open(os.path.join(prj_path, "Inputs", "TRB_inputs.pickle"), "rb") as file:
    (temp, prec, pet, obv_D, obv_M, obv_Y) = pickle.load(file)

#%%
# =============================================================================
# Calibration
# =============================================================================
def cal_batch_indicator(period, target, df_obv, df_sim):
        df_obv = df_obv[period[0]:period[1]]
        df_sim = df_sim[period[0]:period[1]]
        Indicator = HydroCNHS.Indicator()
        df = pd.DataFrame()
        for item in target:
            df_i = Indicator.cal_indicator_df(df_obv[item], df_sim[item],
                                              index_name=item)
            df = pd.concat([df, df_i], axis=0)
        df_mean = pd.DataFrame(df.mean(axis=0), columns=["Mean"]).T
        df = pd.concat([df, df_mean], axis=0)
        return df

def evaluation(individual, info):
    cali_wd, current_generation, ith_individual, formatter, _ = info
    name = "{}-{}".format(current_generation, ith_individual)

    ##### individual -> model
    df_list = cali.Convertor.to_df_list(individual, formatter)
    # ModelDict is from Model Builder (template with -99).
    model = deepcopy(model_dict)
    for i, df in enumerate(df_list):
        s = df_name[i].split("_")[0]
        model = HydroCNHS.load_df_to_model_dict(model, df, s, "Pars")

    ##### Run simuluation
    model = HydroCNHS.Model(model, name)
    try:
        Q = model.run(temp, prec, pet)
    except:
        return (-100,)
    # Get simulation data
    cali_target = ["DLLO", "WSLO"]
    cali_period = ("1981-1-1", "2005-12-31")
    vali_period = ("2006-1-1", "2013-12-31")
    sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)[cali_target]
    sim_Q_D["SHPP"] = model.data_collector.SHPP["Div"]
    sim_Q_D["SCOO"] = model.data_collector.R1["release"]
    cali_target += ["SHPP", "SCOO"]

    sim_Q_M = sim_Q_D[cali_target].resample("MS").mean()
    sim_Q_Y = sim_Q_D[cali_target].resample("YS").mean()

    df_cali_Q_D = cal_batch_indicator(cali_period, cali_target, obv_D, sim_Q_D)
    df_cali_Q_M = cal_batch_indicator(cali_period, cali_target, obv_M, sim_Q_M)
    df_cali_Q_Y = cal_batch_indicator(cali_period, cali_target, obv_Y, sim_Q_Y)

    df_vali_Q_D = cal_batch_indicator(vali_period, cali_target, obv_D, sim_Q_D)
    df_vali_Q_M = cal_batch_indicator(vali_period, cali_target, obv_M, sim_Q_M)
    df_vali_Q_Y = cal_batch_indicator(vali_period, cali_target, obv_Y, sim_Q_Y)

    ##### Save output.txt
    if current_generation == "best":
        with open(os.path.join(cali_wd, "cali_indiv_" + name + ".txt"), 'w') as f:
            f.write("Annual cali/vali result\n")
            f.write(df_cali_Q_Y.round(3).to_csv(sep='\t').replace("\n", ""))
            f.write("\n")
            f.write(df_vali_Q_Y.round(3).to_csv(sep='\t').replace("\n", ""))
            f.write("\n\nMonthly cali/vali result\n")
            f.write(df_cali_Q_M.round(3).to_csv(sep='\t').replace("\n", ""))
            f.write("\n")
            f.write(df_vali_Q_M.round(3).to_csv(sep='\t').replace("\n", ""))
            f.write("\n\nDaily cali/vali result\n")
            f.write(df_cali_Q_D.round(3).to_csv(sep='\t').replace("\n", ""))
            f.write("\n")
            f.write(df_vali_Q_D.round(3).to_csv(sep='\t').replace("\n", ""))
            f.write("\n=========================================================\n")
            f.write("Sol:\n" )
            df = pd.DataFrame(individual, index=cali_inputs["par_name"]).round(4)
            f.write(df.to_string(header=False, index=True))

    fitness = df_cali_Q_M.loc["Mean", "KGE"]
    return (fitness,)

#%%
# cali.helper()
# cali.get_config_template()
# cali.get_inputs_template()

config = {'min_or_max': 'max',
         'pop_size': 200,
         'num_ellite': 1,
         'prob_cross': 0.5,
         'prob_mut': 0.1,
         'stochastic': False,
         'max_gen': 100,
         'sampling_method': 'LHC',
         'drop_record': False,
         'paral_cores': -2,
         'paral_verbose': 1,
         'auto_save': True,
         'print_level': 1,
         'plot': True}

rn_gen = HydroCNHS.create_rn_gen(9)
ga = cali.GA_DEAP(evaluation, rn_gen)
ga.set(cali_inputs, config, formatter, name="Cali_abcd_abm_KGE")
ga.run()
ga.run_individual(ga.solution)

#%%
individual = ga.solution
df_list = cali.Convertor.to_df_list(individual, formatter)
model_best = deepcopy(model_dict)
for i, df in enumerate(df_list):
    s = df_name[i].split("_")[0]
    model = HydroCNHS.load_df_to_model_dict(model_best, df, s, "Pars")
HydroCNHS.write_model(model_best, os.path.join(ga.cali_wd, "Best_abcd_abm_KGE.yaml"))

summary = ga.summary

##### Run simuluation
model = HydroCNHS.Model(model_best, "Best")
Q = model.run(temp, prec, pet)
cali_target = ["DLLO", "WSLO"]
sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)[cali_target]
sim_Q_D["SHPP"] = model.data_collector.SHPP["Div"]
sim_Q_D["SCOO"] = model.data_collector.R1["release"]
cali_target += ["SHPP", "SCOO"]

sim_Q_D["storage"] = model.data_collector.R1["storage"]
cali_target += ["SHPP"]
sim_Q_M = sim_Q_D.resample("MS").mean()
sim_Q_Y = sim_Q_D.resample("YS").mean()

visual = HydroCNHS.Visual()

xy_label_reg = ["Observed streamflow (cms)","Simulated streamflow (cms)"]
xy_label_ts = ["Time","Streamflow (cms)"]
for item in cali_target:
    visual.plot_reg(obv_D[item], sim_Q_D[item], title="Daily_"+item,
                    xy_labal=xy_label_reg)
    visual.plot_reg(obv_M[item], sim_Q_M[item], title="Monthly_"+item,
                    xy_labal=xy_label_reg)
    visual.plot_reg(obv_Y[item], sim_Q_Y[item], title="Annually_"+item,
                    xy_labal=xy_label_reg)
    visual.plot_timeseries(obv_M[[item]], sim_Q_M[[item]],
                           title="Monthly_"+item, xy_labal=xy_label_ts)
    visual.plot_timeseries(obv_Y[[item]], sim_Q_Y[[item]],
                           title="Annually_"+item, xy_labal=xy_label_ts)
#%%
all_indiv = []
for i, v in ga.records.items():
    all_indiv += v
all_indiv_fitness = [i.fitness.values[0] for i in all_indiv]
df_ga = pd.DataFrame(all_indiv)
df_ga["fitness"] = all_indiv_fitness
df_ga = df_ga.drop_duplicates()
df_ga_q99 = df_ga[df_ga["fitness"] > df_ga["fitness"].quantile(0.99)]
stds = df_ga_q99.std()
df_ga_q99.std().mean()
df_ga_q99.std().median()
#df_ga_q99.std().plot.bar()
#stds = df_ga_q99.std()
#df_ga_q99.plot()

