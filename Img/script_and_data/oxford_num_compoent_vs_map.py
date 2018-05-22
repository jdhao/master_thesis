import matplotlib.pyplot as plt
import yaml
import numpy as np
import matplotlib.font_manager as mfm
import matplotlib as mpl

mpl.rcParams.update({'font.size': 15})
font_path = "C:/Windows/fonts/simhei.ttf"
prop = mfm.FontProperties(fname=font_path)


# plt.style.use('seaborn-poster')

with open("data/oxford_crop_query_npc_vs_map_maxpooling_pca_on_paris_conv54_4scale_version2_overlap2.yaml", "r") as f:
    crop_map_paris = yaml.load(f)

with open("data/oxford_full_query_npc_vs_map_maxpooling_pca_on_paris_conv54_4scale_version2_overlap2.yaml", "r") as f:
    full_map_paris = yaml.load(f)

with open("data/oxford_crop_query_npc_vs_map_maxpooling_pca_on_self_conv54_4scale_version2_overlap2.yaml", "r") as f:
    crop_map_self = yaml.load(f)

with open("data/oxford_full_query_npc_vs_map_maxpooling_pca_on_self_conv54_4scale_version2_overlap2.yaml", "r") as f:
    full_map_self = yaml.load(f)


# this is for conv54 features
npc_list = range(16, 520, 16)

# this is for fc6 features
#npc_list = range(128, 4100, 128)

idx_crop_paris = np.argmax(crop_map_paris)
idx_crop_self = np.argmax(crop_map_self)
idx_full_paris = np.argmax(full_map_paris)
idx_full_self = np.argmax(full_map_self)


print("max mAP {}, npc is {} for crop query pca on paris".format(np.max(crop_map_paris), npc_list[idx_crop_paris]))
print("max mAP {}, npc is {} for crop query pca on self".format( np.max(crop_map_self), npc_list[idx_crop_self]))
print("max mAP {}, npc is {} for full query pca on paris".format(np.max(full_map_paris), npc_list[idx_full_paris]))
print("max mAP {}, npc is {} for full query pca on self".format(np.max(full_map_self), npc_list[idx_full_self]))

fig = plt.figure(figsize=(10, 6.18))
ax = fig.add_subplot(111)

colors =['#1b9e77', '#d95f02']
marker1 = 'o'
marker2 = '^'
ax.plot(npc_list, crop_map_paris, color=colors[0],linestyle='solid', marker=marker1, label= 'crop-paris')
ax.plot(npc_list, crop_map_self, color=colors[1],linestyle='solid', marker=marker1,  label = 'crop-self')
#ax.plot(npc_list, crop_map_ukb, color='b', linestyle='solid', marker=marker1, label = 'crop-ukb')
ax.plot(npc_list, full_map_paris, color=colors[0],linestyle='solid', marker=marker2, label = 'full-paris')
ax.plot(npc_list, full_map_self, color=colors[1],linestyle='solid', marker=marker2, label= 'full-self')
#ax.plot(npc_list, full_map_ukb, color='b', linestyle='solid', marker=marker2, label = 'full-ukb')

ax.set_ylabel("mAP")
ax.set_xlabel("特征维度", fontproperties=prop)
# ax.set_xticks([x for x in range(128, 4100, 384)])
ax.set_xticks([x for x in range(16, 520, 64)])
ax.set_yticks([x for x in np.arange(0.34, 0.76, 0.1)])
ax.legend(loc="lower right")
ax.grid()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', top='off', right='off')

plt.savefig("chapter_mfc_oxford_pca_other_self.pdf", bbox_inches='tight')
plt.show()
