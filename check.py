import numpy as np
import matplotlib.pyplot as plt

ts = np.load('data/raw/time_series_fold.npy')
x = ts[0]

plt.figure(figsize=(10, 4))
plt.plot(x)
plt.title("Fold Bifurcation Time Series")
plt.show()

window = 50
rolling_var = [np.var(x[i:i+window]) for i in range(len(x) - window)]

plt.figure(figsize=(10, 4))
plt.plot(rolling_var)
plt.title("Rolling Variance (EWS)")
plt.show()
