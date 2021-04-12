
import os
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rc('font', family='serif')
# mpl.rcParams['font.family'] = 'Inconsolata'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 0.5

SCALE_CPU = True
RESULTS_DIR_CPU = "results/V100/hessian_cpu/"
RESULTS_DIR_GPU = "results/"
ARCHS = ["V100", "A100"]
SUBDIR = "/batch_hessian/"

fig, ax = plt.subplots(figsize=(8, 5), ncols=len(ARCHS), sharey=True)

for (k, arch) in enumerate(ARCHS):
    localdir = RESULTS_DIR_GPU + arch + SUBDIR
    RESULTS = glob.glob(localdir + "/*")

    for results in RESULTS:
        name = results.split("/")[-1]
        print(name)
        output = np.loadtxt(results)
        batches = output[:, -1]
        # Parallel scaling
        myfilter = batches > 0.0
        batches = batches[myfilter]

        if SCALE_CPU:
            output_cpu = np.loadtxt(RESULTS_DIR_CPU + name)
            time_cpu = output_cpu[1]
            timings = output[myfilter, 1] / time_cpu
            p = ax[k].plot(batches, timings, "-o", label=name)
            color = p[0].get_color()
            # Linear scaling
            t0 = output[0, 1] / time_cpu
            linear_scaling = t0 / batches * batches[0]
            ax[k].plot(batches, linear_scaling, ":", lw=.8, c=color, alpha=.7)
        else:
            timings = output[myfilter, 1]
            p = ax[k].plot(batches, timings, "-o", label=name)
            color = p[0].get_color()
            # Linear scaling
            t0 = output[0, 1]
            linear_scaling = t0 / batches * batches[0]
            ax[k].plot(batches, linear_scaling, ":", lw=.8, c=color, alpha=.7)


    ax[k].grid(ls=":")
    ax[k].set_yscale("log", base=10)
    ax[k].set_xscale("log", base=2)
    ax[k].set_xlabel("#batches")
    ax[k].set_title(arch)

if SCALE_CPU:
    ax[0].set_ylabel("Ratio (GPU / CPU)")
else:
    ax[0].set_ylabel("Time (s)")
ax[0].legend(fontsize="x-small", loc=3)
plt.tight_layout()
plt.savefig("batch_hessian.pdf")
plt.show()
