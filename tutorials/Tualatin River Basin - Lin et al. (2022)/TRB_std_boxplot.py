import os
import matplotlib.pyplot as plt
import pandas as pd
import HydroCNHS.calibration as cali

##### Path and Load Model Test
# Get this file directory.
prj_path, this_filename = os.path.split(__file__)


def get_stds(model):
    def evaluation():
        pass

    # Collect GA evaluation from three random seeds.
    df_ga_all = pd.DataFrame()
    for seed in [5, 10, 13]:
        save_ga_file_path = os.path.join(
            prj_path, "Cali_{}_abm_KGE_{}/GA_auto_save.pickle".format(model, seed)
        )
        ga = cali.GA_DEAP(evaluation)
        ga.load(save_ga_file_path, "")
        all_indiv = []
        for i, v in ga.records.items():
            all_indiv += v
        all_indiv_fitness = [i.fitness.values[0] for i in all_indiv]
        df_ga = pd.DataFrame(all_indiv)
        df_ga["fitness"] = all_indiv_fitness
        df_ga = df_ga.drop_duplicates()
        df_ga_all = pd.concat([df_ga_all, df_ga])
        print("[{}] Seed {}: {}".format(model, seed, max(df_ga["fitness"])))
    df_ga_q99 = df_ga_all[df_ga_all["fitness"] > df_ga_all["fitness"].quantile(0.95)]
    stds = df_ga_q99.std()
    return list(stds)[:-1]


# %%
# Get stds of calibrated parameters (top 1% fitness)
data = [get_stds("gwlf"), get_stds("abcd")]

# Plot boxplot
fig, ax = plt.subplots()
ax.set_ylabel("Standard deviation", fontsize=13)
ax.set_xticklabels(["M$_{gwlf}$", "M$_{abcd}$"])
ax.boxplot(data)
plt.show()
