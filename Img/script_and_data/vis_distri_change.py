import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as mfm
import matplotlib as mpl

mpl.rcParams.update({'font.size': 15})
font_path = "C:/Windows/fonts/simhei.ttf"
prop = mfm.FontProperties(fname=font_path)

dist_pos1 = np.load("data/dist_pos_vgg16_full.npy")
dist_neg1 = np.load("data/dist_neg_vgg16_full.npy")

dist_pos2 = np.load("data/dist_pos_cls_run4_model_full.npy")
dist_neg2 = np.load("data/dist_neg_cls_run4_model_full.npy")

dist_pos3 = np.load("data/dist_pos_retr_from_cls_run10_full.npy")
dist_neg3 = np.load("data/dist_neg_retr_from_cls_run10_full.npy")


num_bin = 500
# colors = ['#1b9e77', '#d95f02']
colors = ["#bebebe", "#3a3a3a"]
fig, (ax1, ax2, ax3)= plt.subplots(nrows=3, ncols=1, figsize=(10, 6))


ax1.hist(dist_pos1, bins=num_bin, normed=True,
         color=colors[0], label='正样本对')
ax1.hist(dist_neg1, bins=num_bin, normed=True,
         color=colors[1], alpha=0.8, label='负样本对')

ax1.set_xlim([0.35, 1.43])
xleft, xright = ax1.get_xlim()
ybottom, ytop = ax1.get_ylim()
ax1.set_aspect(abs((xright-xleft)/(ybottom-ytop))*0.2)

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc='lower left', bbox_to_anchor=(-0.02, 0.92),
                    fontsize=15, ncol=2, frameon=False, prop=prop)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='both', top='off', right='off')

for item in ([ax1.xaxis.label, ax1.yaxis.label] +
             ax1.get_xticklabels() + ax1.get_yticklabels()):
    item.set_fontsize(15)

ax2.hist(dist_pos2, bins=num_bin, normed=True,
         color=colors[0])
ax2.hist(dist_neg2, bins=num_bin, normed=True,
         color=colors[1], alpha=0.8)

ax2.set_xlim([0.35, 1.43])
xleft, xright = ax2.get_xlim()
ybottom, ytop = ax2.get_ylim()
ax2.set_aspect(abs((xright-xleft)/(ybottom-ytop))*0.2)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='both', top='off', right='off')

for item in ([ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
    item.set_fontsize(15)

ax3.hist(dist_pos3, bins=num_bin, normed=True,
         color=colors[0])
ax3.hist(dist_neg3, bins=num_bin, normed=True,
         color=colors[1], alpha=0.8)

ax3.set_xlim([0.35, 1.43])
xleft, xright = ax3.get_xlim()
ybottom, ytop = ax3.get_ylim()
ax3.set_aspect(abs((xright-xleft)/(ybottom-ytop))*0.2)
ax3.set_xlabel('图像对特征之间的距离', fontproperties=prop)

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.tick_params(axis='both', top='off', right='off')

for item in ([ax3.xaxis.label, ax3.yaxis.label] +
             ax3.get_xticklabels() + ax3.get_yticklabels()):
    item.set_fontsize(15)

# legend = plt.figlegend(handles, labels, loc='lower left', ncol=2, frameon=False,
#               bbox_to_anchor=(0.12, 0.88), fontsize='large')

# plt.tight_layout(rect=[0,0,1, 0.9])

plt.savefig("chapter_double_margin_dist_distribution_change.pdf", bbox_inches='tight')
# plt.savefig("dist_distribution_change.jpg", bbox_extra_artists=(legend,),
#             bbox_inches='tight')

plt.show()
