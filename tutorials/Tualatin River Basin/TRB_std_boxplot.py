import os
import matplotlib.pyplot as plt
import pandas as pd
import HydroCNHS.calibration as cali

##### Path and Load Model Test
# Get this file directory.
prj_path, this_filename = os.path.split(__file__)

def get_stds(save_ga_file_path):
    def evaluation():
        pass
    ga = cali.GA_DEAP(evaluation)
    ga.load(save_ga_file_path, "")
    all_indiv = []
    for i, v in ga.records.items():
        all_indiv += v
    all_indiv_fitness = [i.fitness.values[0] for i in all_indiv]
    df_ga = pd.DataFrame(all_indiv)
    df_ga["fitness"] = all_indiv_fitness
    df_ga = df_ga.drop_duplicates()
    df_ga_q99 = df_ga[df_ga["fitness"] > df_ga["fitness"].quantile(0.99)]
    stds = df_ga_q99.std()
    return list(stds)[:-1]
#%%

# GA_auto_save path
save_ga_file_gwlf = os.path.join(prj_path, "Cali_gwlf_abm_KGE/GA_auto_save.pickle")
save_ga_file_abcd = os.path.join(prj_path, "Cali_abcd_abm_KGE/GA_auto_save.pickle")

# Get stds of calibrated parameters (top 1% fitness)
data = [get_stds(save_ga_file_gwlf),
        get_stds(save_ga_file_abcd)]

# Plot botplot
fig, ax = plt.subplots()
ax.set_ylabel("Standard deviation", fontsize=13)
ax.set_xticklabels(["M$_{gwlf}$","M$_{abcd}$"])
ax.boxplot(data)
plt.show()