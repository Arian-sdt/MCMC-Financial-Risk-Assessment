import pandas as pd
import numpy as np

data = pd.read_csv("GSPC_clean.csv")

x = data["LogReturn"].astype(float).to_numpy()

print("mean(LogReturn):", np.mean(x))
print("std(LogReturn):",  np.std(x))
print("var(LogReturn):",  np.var(x))