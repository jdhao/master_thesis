import matplotlib.pyplot as plt
import json
import matplotlib.font_manager as mfm
import matplotlib as mpl

mpl.rcParams.update({'font.size': 15})
font_path = "C:/Windows/fonts/simhei.ttf"
prop = mfm.FontProperties(fname=font_path)

with open('data/train_val_test_cls_num.json', 'r') as f:
    train_val_test_info = json.load(f)

train_cls_info = train_val_test_info['train']
val_cls_info = train_val_test_info['validation']
test_cls_info = train_val_test_info['test']

num = list(train_cls_info.values()) + list(val_cls_info.values()) +\
      list(test_cls_info.values())
# random.shuffle(num)
num.sort()

# plt.style.use('seaborn-poster')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
# ax.bar(range(len(num)), num, width=0.5, color='#2c7fb8')
ax.plot(range(len(num)), num)

ax.set_yticks(range(20, 262, 40))
ax.grid(which='major', axis='y', linestyle='--')
ax.set_xlabel('类别编号', fontproperties=prop)
ax.set_ylabel('图片数量', fontproperties=prop)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', top='off', right='off')

xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*0.4)

plt.savefig('dataset_cls_img_num.pdf', bbox_inches='tight')
plt.show()
