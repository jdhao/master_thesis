import json
import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib as mpl
import matplotlib.font_manager as mfm


# mpl.rcParams['pdf.fonttype'] = 42

mpl.rcParams.update({'font.size': 15})
font_path = "C:/Windows/fonts/simhei.ttf"
prop = mfm.FontProperties(fname=font_path)

xlabel = "类别编号"
ylabel1 = "总偏差"
ylabel2 = "个人偏差"
markers = ['o', '^', '<', 'x', 's']

color = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33",
         "#a65628", "#f781bf"]

with open("data/sorted_deviation.json", "r") as f:
    sorted_overall_info = json.load(f)

with open("data/personal_deviation.json", "r") as f:
    personal_deviation = json.load(f)

overall_deviation = np.zeros(len(sorted_overall_info))

for i, (cls_name, value) in enumerate(sorted_overall_info):
    overall_deviation[i] = value

sorted_personal_deviation = np.zeros((5, len(personal_deviation)))

for i, data in enumerate(sorted_overall_info):
    sorted_personal_deviation[:, i] = personal_deviation[data[0]]


# smooth the curves
for i in range(5):
    sorted_personal_deviation[i] = savgol_filter(sorted_personal_deviation[i], 59, 3)

# plt.style.use('ggplot')
fig = plt.figure(figsize=(10, 6))

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, sharex=ax1)

ax1.plot(overall_deviation, color=color[6])
ax1.set_ylim([overall_deviation.min() * 0.9, overall_deviation.max() * 1.1])

# ax1.hlines(np.arange(0.2, 0.72, 0.1), xmin=0, xmax=180, colors='grey',
# alpha=0.2)
ax1.set_xlabel(xlabel, fontproperties=prop)
ax1.set_ylabel(ylabel1, fontproperties=prop)

for i in range(5):
    ax2.plot(sorted_personal_deviation[i],
             linewidth=2,
             color=color[i],
             label='评分者 {}'.format(i))

ax2.set_ylim([sorted_personal_deviation.min() * 0.9, sorted_personal_deviation.max() * 1.1])

for ax in [ax1, ax2]:
    # ax.tick_params('both', length=0, width=0, which='major')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.set_aspect((xmax-xmin)/(ymax-ymin), adjustable='box-forced')

ax2.set_xlabel(xlabel, fontproperties=prop)
ax2.set_ylabel(ylabel2, fontproperties=prop)

leg2 = ax2.legend(fancybox=True, labelspacing=0.2, prop=prop)
leg2.get_frame().set_alpha(0.8)

[line.set_linewidth(3) for line in leg2.get_lines()]
[text.set_fontsize('small') for text in leg2.get_texts()]

plt.subplots_adjust(wspace=0.3)
plt.savefig("overall_individual_devi.pdf", bbox_inches='tight')
plt.show()

"""
fig2 = plt.figure(figsize=(10,6))
ax3 = fig2.add_subplot(111)

for i in range(5):
    ax3.plot(sorted_personal_deviation[i],
             linewidth=2,
             color=color[i],
             label='labeler {}'.format(i))
ax3.set_xlabel('class index')
ax3.set_ylabel('personal deviation')
ax3.legend()
plt.savefig("../results/invidual_devi.jpg", bbox_inches='tight')

plt.show()
"""
