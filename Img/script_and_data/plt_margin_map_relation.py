import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'

colors = ["#e6194b",
          "#3cb44b",
          "#ffe119",
          "#0082c8",
          "#f58231",
          "#911eb4"]

margin1 = [0.5, 0.6, 0.7, 0.8, 0.9]

map1 = np.array([0.658, 0.674, 0.680, 0.679, 0.672])
map2 = np.array([0.661, 0.670, 0.676, 0.684, 0.677])
map3 = np.array([0.662, 0.664, 0.669, 0.673, 0.666])

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(margin1, map1*100, '-o', color=colors[0], markersize=10, lw=3.5,
        label=r'$\alpha_2 = 1.1$')
ax.plot(margin1, map2*100, '-s', color=colors[1], markersize=10, lw=3.5,
        label=r'$\alpha_2 = 1.2$')
ax.plot(margin1, map3*100, '-^', color=colors[3], markersize=10, lw=3.5,
        label=r'$\alpha_2 = 1.3$')

# ax.plot(1.0, 66.0, '*', color=colors[4])

ax.set_xticks(np.arange(0.5, 0.95, 0.1))
ax.set_yticks(np.arange(65.5, 69, 0.5))

ax.set_xlabel(r'$\alpha_1$')
ax.set_ylabel('mAP (%)')
ax.legend()
ax.grid(which='major', axis='both', linestyle='--')

for item in ([ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels() +
             ax.legend().get_texts()):
    item.set_fontsize(18)

xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*0.6)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', top='off', right='off')

plt.savefig("chapter_double_margin_impact_of_margins.pdf", bbox_inches='tight',
            pad_inches=0)

plt.show()
