import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as mfm
import matplotlib as mpl

mpl.rcParams.update({'font.size': 15})
mpl.rcParams['mathtext.fontset'] = 'cm'
font_path = "C:/Windows/fonts/simhei.ttf"
prop = mfm.FontProperties(fname=font_path)

# plt.style.use('ggplot')
# colors = ["#e41a1c",
#           "#377eb8",
#           "#4daf4a",
#           "#984ea3",
#           "#ff7f00",
#           "#ffff33"]

# colors = ["#1b9e77",
#          "#d95f02",
#          "#7570b3",
#          "#e7298a",
#          "#66a61e",
#          "#e6ab02"]

# colors = ["#8772c9",
#             "#70a845",
#             "#c8598f",
#             "#4dad97",
#             "#cc5a43",
#             "#b69040"]

colors = ["#e6194b",
          "#3cb44b",
          "#ffe119",
          "#0082c8",
          "#f58231",
          "#911eb4"]

markers = ['o', '^', 's', 'x']

# neural codes
m1 = np.array([ 0.6125, 0.775, 0.8375, 0.875, 0.9125, 0.9875])

# spoc
m2 = np.array([0.6750, 0.8125, 0.8625, 0.9000, 0.9375, 0.9750])

# MAC (maximum activation of covolutions)
m3 = np.array([0.8375, 0.9125, 0.9500, 0.9625, 0.9875, 0.9875])

# MFC
m4 = np.array([0.8125, 0.8500, 0.8625, 0.9500, 0.9625, 0.9875])

# cnn retrieval learns from bow
m5 = np.array([0.8000, 0.8625, 0.9250, 0.9625, 0.9625, 0.9750])

# triplet network (deep image retrieval)
m6 = np.array([ 0.9375, 0.9625, 0.9875, 1., 1., 1.])

# ours double margin, no cls fine-tune
# m4 = np.array([0.875, 0.8875, 0.9500, 0.9750, 1.0, 1.0])
# ours cls
# m5 = np.array([ 0.925, 0.9625, 0.975, 0.9875, 1., 1])
# ours double margin + cls
m7 = np.array([ 0.95, 0.9875, 0.9875, 1., 1., 1.])

position = [1, 2, 4, 8, 16, 32]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(position, m1*100, linestyle='-', color=colors[0], marker=markers[0],
        label="NeuralCode")

ax.plot(position, m2*100, linestyle='-', color=colors[1], marker=markers[1],
        label="SPoC")

ax.plot(position, m3*100, linestyle='-', color=colors[2], marker=markers[2],
        label="MAC")

ax.plot(position, m4*100, linestyle='-', color=colors[3], marker=markers[3],
        label="MFC")

ax.plot(position, m5*100, linestyle='--', color=colors[0], marker=markers[0],
        label="Siamese-MAC")


ax.plot(position, m6*100, linestyle='--', color=colors[1], marker=markers[1],
        label="Triplet")

ax.plot(position, m7*100, linestyle='--', color=colors[2], marker=markers[2],
        label="Ours (Retr+Cls)")


ax.set_xscale('log')
ax.set_xticks(position)
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

# ax.set_xticklabels(position)
# ax.tick_params(axis='x', which='minor', bottom='off')
ax.minorticks_off()
ax.grid(linestyle='--')
ax.set_xlabel(r"$K$")
ax.set_ylabel("Rank-k 准确率(%)", fontproperties=prop)

ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', top='off', right='off')
plt.savefig("chapter_double_margin_compare_rank_k_accu.pdf", bbox_inches='tight',
            pad_inches=0)
plt.show()
