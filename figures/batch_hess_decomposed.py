
import os
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rc('font', family='serif')
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 0.5
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

RESULTS_DIR = "results/"
ARCHS = ["V100", "A100"]
SUBDIR = "/batch_hessprod/"
scale = True

fig, ax = plt.subplots(figsize=(8, 5), ncols=2, sharey=True)


for (k, arch) in enumerate(ARCHS):
    localdir = RESULTS_DIR + arch + SUBDIR + "case9241pegase.m"

    output = np.loadtxt(localdir)
    scale_factor = 1e3

    batches = 1
    time_mul1 = output[:, 1] / batches * scale_factor
    time_sp1 = output[:, 2] / batches * scale_factor
    time_ad = output[:, 3] / batches * scale_factor
    time_sp2 = output[:, 4] / batches * scale_factor
    time_mul2 = output[:, 5] / batches * scale_factor

    batches = output[:, -1]
    ind = [i for i in range(len(batches))]

    p1 = ax[k].bar(ind, time_sp1, 0.55, label='Forward solve', color="darkblue")
    p2 = ax[k].bar(ind, time_ad, 0.55, label='AutoDiff', color="darkred", bottom=time_sp1)
    bot = time_sp1 + time_ad
    p3 = ax[k].bar(ind, time_sp2, 0.55, label='Backward solve', color="darkgreen", bottom=bot)

    ax[k].axhline(0, color='grey', linewidth=0.8)
    ax[k].set_title(arch)

    ax[k].set_xticks(ind)
    ax[k].set_xticklabels([int(i) for i in batches])

    # ax.set_yscale("log", base=10)
    ax[k].set_xlabel('#Batches')
    ax[k].grid(ls=":")


ax[-1].legend(fontsize="small")
ax[0].set_ylabel('Time (ms)')
plt.savefig("time_decomposed_batch_hessprod.pdf")
plt.tight_layout()
plt.show()
