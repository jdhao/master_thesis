import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter

import matplotlib.font_manager as mfm
import matplotlib as mpl

mpl.rcParams.update({'font.size': 15})
font_path = "C:/Windows/fonts/simhei.ttf"
prop = mfm.FontProperties(fname=font_path)

eps = 0.00001

loss_val_s = np.load("data/loss_value_epoch1_single_margin_from_ori_vgg.npz")
loss_val_d = np.load("data/loss_value_epoch1_double_margin_from_ori_vgg.npz")

pos_loss_s = loss_val_s['pos']
neg_loss_s = loss_val_s['neg']
pos_loss_d = loss_val_d['pos']
neg_loss_d = loss_val_d['neg']

loss_contrib_s = []
loss_contrib_d = []

for i in range(pos_loss_s.size):
    unit_loss_pos = sum(pos_loss_s[i])/len(pos_loss_s[i])

    neg_loss_real = [x for x in neg_loss_s[i] if abs(x) > eps]
    unit_loss_neg = sum(neg_loss_real)/len(neg_loss_real)

    pos_ratio = unit_loss_pos / (unit_loss_pos + unit_loss_neg)

    loss_contrib_s.append(pos_ratio)

for i in range(pos_loss_s.size):
    unit_loss_pos = sum(pos_loss_d[i])/len(pos_loss_d[i])

    neg_loss_real = [x for x in neg_loss_d[i] if abs(x) > eps]
    unit_loss_neg = sum(neg_loss_real)/len(neg_loss_real)

    pos_ratio = unit_loss_pos / (unit_loss_pos + unit_loss_neg)

    loss_contrib_d.append(pos_ratio)

loss_contrib_s = np.array(loss_contrib_s)*100
loss_contrib_d = np.array(loss_contrib_d)*100
# loss_contrib_s = savgol_filter(loss_contrib_s, 9, 2)

fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(10, 7))
ax1.plot(loss_contrib_s, label='正样本对')
ax1.plot(100-loss_contrib_s, label='负样本对')
ax1.set_ylabel("对总损失的贡献（%）", fontproperties=prop)

xleft, xright = ax1.get_xlim()
ybottom, ytop = ax1.get_ylim()
ax1.set_aspect(abs((xright-xleft)/(ybottom-ytop))*0.25)

handles, labels = ax1.get_legend_handles_labels()
leg = ax1.legend(handles, labels, loc='lower left', bbox_to_anchor=(-0.02, 0.95),
                    fontsize=12, ncol=2, frameon=False, prop=prop)
for line in leg.get_lines():
    line.set_linewidth(3.0)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='both', top='off', right='off')


ax2.plot(loss_contrib_d)
ax2.plot(100-loss_contrib_d)

ax2.set_xlabel("批（batch）编号", fontproperties=prop)
ax2.set_ylabel("对总损失的贡献（%）", fontproperties=prop)

xleft, xright = ax2.get_xlim()
ybottom, ytop = ax2.get_ylim()
ax2.set_aspect(abs((xright-xleft)/(ybottom-ytop))*0.25)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='both', top='off', right='off')

plt.subplots_adjust(hspace=0.1)
plt.savefig("chapter_double_margin_pos_neg_loss_contrib.pdf", bbox_inches='tight')
plt.show()
