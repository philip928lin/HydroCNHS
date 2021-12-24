import os
import matplotlib.pyplot as plt
import pandas as pd

path = r"C:\Users\ResearchPC\OneDrive\Lehigh\0_Proj2_HydroCNHS\Model\Stds.csv"
df = pd.read_csv(path)

data = [list(df["Mgwlf"]), list(df["Mabcd"].dropna())]
fig, ax = plt.subplots()
ax.set_ylabel("Standard deviation", fontsize=13)
ax.set_xticklabels(["M$_{gwlf}$","M$_{abcd}$"])
ax.boxplot(data)

plt.show()