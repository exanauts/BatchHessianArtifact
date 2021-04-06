
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results/batch_hessian/autodiff/"
RESULTS = glob.glob(RESULTS_DIR + "/*")

x_b = [4, 8, 16, 32, 64, 128]
fig, ax = plt.subplots(figsize=(8, 5))
for results in RESULTS:
    name = results.split("/")[-1]
    nres = np.loadtxt(results)
    nres /= 1e6
    ax.errorbar(x_b, nres[:, 1], yerr=3 * nres[:, 2] / np.sqrt(10000), label=name, fmt="-o")

ax.grid(ls=":")
ax.set_yscale("log", base=10)
ax.set_xscale("log", base=2)
ax.set_ylabel("Time (ms)")
ax.set_xlabel("#batches")
ax.legend()

plt.savefig("batch_autodiff.pdf")
plt.show()
