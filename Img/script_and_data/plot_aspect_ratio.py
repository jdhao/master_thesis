import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as mfm

mpl.rcParams.update({'font.size': 15})

font_path = "C:/Windows/fonts/simhei.ttf"
prop = mfm.FontProperties(fname=font_path)

ratio1 = np.load('data/firearm_aspect_ratio_rotation.npy')
ratio2 = np.load('data/paris_aspect_ratio.npy')
ratio3 = np.load('data/oxford_aspect_ratio.npy')
ratio4 = np.load('data/imagenet_aspect_ratio.npy')

inv_ratio1 = 1.0 / ratio1
inv_ratio2 = 1.0 / ratio2
inv_ratio3 = 1.0 / ratio3
inv_ratio4 = 1.0 / ratio4

ax_ratio = 0.6
num_bin = 40

colors = ['#1b9e77', '#d95f02']
fig = plt.figure(figsize=(10, 6))
ax1= fig.add_subplot(221)

ax1.hist(ratio1, bins=num_bin, histtype='stepfilled', normed=True,
         color=colors[0], label=r'$\frac{W}{H}$')
ax1.hist(inv_ratio1, bins=num_bin, histtype='stepfilled', normed=True,
         color=colors[1], alpha=0.6, label=r'$\frac{H}{W}$')

ax1.set_xlim([0, 5])
xleft, xright = ax1.get_xlim()
ybottom, ytop = ax1.get_ylim()
ax1.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ax_ratio)

# ax1.set_title("Firearm14k")
ax1.text(0.8, 0.8, "Firearm14k", ha="center", va="center",
	transform=ax1.transAxes)
# ax1.set_xlabel("Aspect ratio")
ax1.set_ylabel("概率密度", fontproperties=prop)
# ax1.legend(loc='best')

ax1.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01),
	ncol=2, borderaxespad=0, frameon=False)


ax2 = fig.add_subplot(222)
ax2.hist(ratio2, bins=num_bin, histtype='stepfilled', normed=True,
         color=colors[0], label='W/H')
ax2.hist(inv_ratio2, bins=num_bin, histtype='stepfilled', normed=True,
         color=colors[1], alpha=0.6, label='H/W')
ax2.set_xlim([0, 5])
xleft, xright = ax2.get_xlim()
ybottom, ytop = ax2.get_ylim()
ax2.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ax_ratio)

# ax2.set_xlabel("Aspect ratio")
# ax2.set_ylabel("Frequency")
# ax2.set_title("Paris6k")
ax2.text(0.8, 0.8, "Paris6k", ha="center", va="center",
	transform=ax2.transAxes)
# ax2.legend(loc='best')

ax3 = fig.add_subplot(223)
ax3.hist(ratio3, bins=num_bin, histtype='stepfilled', normed=True,
         color=colors[0], label='W/H')
ax3.hist(inv_ratio3, bins=num_bin, histtype='stepfilled', normed=True,
         color=colors[1], alpha=0.6, label='H/W')
ax3.set_xlim([0, 5])
xleft, xright = ax3.get_xlim()
ybottom, ytop = ax3.get_ylim()
ax3.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ax_ratio)

ax3.set_xlabel("长宽的比例", fontproperties=prop)
ax3.set_ylabel("概率密度", fontproperties=prop)
ax3.text(0.8, 0.8, "Oxford5k", ha="center", va="center",
	transform=ax3.transAxes)

# ax3.set_title("Oxford5k")
# ax3.legend(loc='best')

ax4 = fig.add_subplot(224)
ax4.hist(ratio4, bins=num_bin, histtype='stepfilled', normed=True,
         color=colors[0], label='W/H')
ax4.hist(inv_ratio4, bins=num_bin, histtype='stepfilled', normed=True,
         color=colors[1], alpha=0.6, label='H/W')
ax4.set_xlim([0, 5])
xleft, xright = ax4.get_xlim()
ybottom, ytop = ax4.get_ylim()
ax4.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ax_ratio)

ax4.set_xlabel("长宽的比例", fontproperties=prop)
ax4.text(0.8, 0.8, "ImageNet", ha="center", va="center",
	transform=ax4.transAxes)
# ax4.set_ylabel("Frequency")
# ax4.set_title("ImageNet")
# ax4.legend(loc='best')
plt.subplots_adjust(hspace=0.2)

plt.savefig("dataset_aspect_ratio.pdf", bbox_inches='tight')
plt.show()
