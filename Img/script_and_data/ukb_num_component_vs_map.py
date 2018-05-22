import matplotlib.pyplot as plt
import numpy as np
import yaml
import matplotlib.font_manager as mfm
import matplotlib as mpl

mpl.rcParams.update({'font.size': 15})
font_path = "C:/Windows/fonts/simhei.ttf"
prop = mfm.FontProperties(fname=font_path)

#for conv54
pc_list = range(64, 520, 32)
# plt.style.use("seaborn-poster")

# for fc6
# pc_list = range(128, 4100, 128)

with open("data/npc_vs_accuracy_maxpooling_pca_on_self_conv54_4scale_version2_overlap2.yaml", "r") as f:
    accuracy_list_self = yaml.load(f)
with open("data/npc_vs_accuracy_maxpooling_pca_on_oxford_conv54_4scale_version2_overlap2.yaml", "r") as f:
    accuracy_list_oxford = yaml.load(f)
# with open("npc_vs_accuracy_maxpooling_pca_on_self_fc6_3scale_overlap.yaml", "r") as f:
    # accuracy_list_self = yaml.load(f)

idx_max_self = np.argmax(accuracy_list_self)
idx_max_oxford = np.argmax(accuracy_list_oxford)

print("max accuracy {}, npc is {} for pca on self".format(np.max(accuracy_list_self), pc_list[idx_max_self]))
print("max accuracy {}, npc is {} for pca on oxford".format(np.max(accuracy_list_oxford), pc_list[idx_max_oxford]))

colors =['#1b9e77', '#d95f02']

fig = plt.figure(figsize=(10, 6.18))
ax = fig.add_subplot(111)
ax.plot(pc_list, accuracy_list_self,  '-o', color = colors[0],label='pca-self')
ax.plot(pc_list, accuracy_list_oxford,'-^', color=colors[1],label='pca-oxford')
ax.set_ylabel("准确率", fontproperties=prop)
ax.set_xlabel("特征维度", fontproperties=prop)
ax.set_xticks([x for x in range(64, 520, 64)])
ax.set_yticks([x for x in np.arange(3.58, 3.80, 0.04)])

ax.legend(loc="lower right")
ax.grid()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', top='off', right='off')
plt.savefig("chapter_mfc_ukb_pca_other_self.pdf", bbox_inches='tight')
plt.show()
