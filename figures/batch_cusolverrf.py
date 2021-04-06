
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results/cusolver/"

scale = True
scale_ms = 1e6
npres = np.loadtxt(RESULTS_DIR + "case300.txt")

x_b = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
ind = [i + 1 for i in range(len(x_b))]

ref = npres[1, 1] / scale_ms
time_median = npres[2:, 1] / scale_ms
time_std = npres[2:, 2] / scale_ms
n_samples = npres[2:, 3]
y_err = time_std / n_samples

fig, ax = plt.subplots(figsize=(8, 5))

p0 = ax.bar(0, ref, 0.55, label='UMFPACK', color="darkred")
if scale:
    p1 = ax.bar(ind, time_median / x_b, 0.55, yerr=y_err / x_b, label='CUSOLVER_RF', color="darkblue")
else:
    p1 = ax.bar(ind, time_median, 0.55, yerr=y_err, label='CUSOLVER_RF', color="darkblue")

ax.axhline(0, color='grey', linewidth=0.8)
ax.set_title('Performance of CUSOLVER_RF on case300')

x_b.insert(0, 1)
ax.set_xticks(range(len(x_b)))
ax.set_xticklabels(x_b)

ax.set_ylabel('Time per RHS (ms)')
ax.set_yscale("log", base=10)
ax.set_xlabel('#RHS')
ax.grid(ls=":")
ax.legend()

plt.savefig("batch_cusolver_2.pdf")
plt.show()
