
import os
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rc('font', family='serif')
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 0.5

RESULTS_DIR = "results/"
SUBPATH = "/batch_autodiff/"
ARCHS = ["V100", "A100"]


batches = [4, 8, 16, 32, 64, 128, 256, 512]
fig, ax = plt.subplots(figsize=(8, 5), ncols=len(ARCHS), sharey=True)
for (k, arch) in enumerate(ARCHS):
    result_path = RESULTS_DIR + arch + SUBPATH
    RESULTS = glob.glob(result_path + "/*")
    for results in RESULTS:
        name = results.split("/")[-1]

        output = np.loadtxt(results)
        # Parallel scaling
        time_median = output[:, 1] / 1e6 / batches

        p = ax[k].plot(batches, time_median, "-o", label=name)
        color = p[0].get_color()

        # Linear scaling
        t0 = output[0, 1] / 1e6
        linear_scaling = t0 / batches
        ax[k].plot(batches, linear_scaling, ":", lw=.8, c=color, alpha=.7)

    ax[k].grid(ls=":")
    ax[k].set_yscale("log", base=10)
    ax[k].set_xscale("log", base=2)
    ax[k].set_xlabel("#batches")
    ax[k].set_title(arch)

ax[0].set_ylabel("Time per batch (ms)")
ax[0].legend(fontsize="x-small", loc=3)

plt.savefig("comparison_autodiff.pdf")
plt.tight_layout()
plt.show()
