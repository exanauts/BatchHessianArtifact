
import os
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rc('font', family='serif')
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 0.5

RESULTS_DIR = "results/V100/cusolver/"

scale = True
scale_ms = 1e6

batches = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

RESULTS = glob.glob(RESULTS_DIR + "/*")

fig, ax = plt.subplots(figsize=(8, 5))

for results in RESULTS:
    name = results.split("/")[-1]
    name = name.split(".")[0]

    output = np.loadtxt(results)
    # Parallel scaling
    time_cpu = output[1, 1] / scale_ms
    time_median = output[2:, 1] / scale_ms / batches

    p = ax.plot(batches, time_median, "-o", label=name + " cusolverRF")
    color = p[0].get_color()

    # Linear scaling
    t0 = output[2, 1] / scale_ms
    linear_scaling = t0 / batches
    ax.plot(batches, linear_scaling, ":", lw=.8, c=color, alpha=.7)

    ax.hlines(time_cpu, 0, batches[-1], linestyle=":", lw=2.0, alpha=.9, colors=color, label=name + " UMFPACK")

# x_b.insert(0, 1)
ax.set_xticks(batches)
ax.set_xticklabels(batches)

ax.set_ylabel('Time per RHS (ms)')
ax.set_yscale("log", base=10)
ax.set_xscale("log", base=2)
ax.set_xlabel('#Batches')
ax.grid(ls=":")
ax.legend(loc=3, fontsize="x-small")

plt.savefig("batch_cusolver.pdf")
plt.tight_layout()
plt.show()
