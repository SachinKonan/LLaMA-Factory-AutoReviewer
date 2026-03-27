import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

mpl.rcParams.update({
"text.usetex": True,
"font.family": "sans-serif",
"font.sans-serif": "helvetica",
})
labelsize = 20
titlesize = 20
legendsize=20
fontweight = 20
ticksize=16

res_simple_metric = [7.6565,7.5418,7.3663,7.3560,7.3172,7.2945,7.241, 7.19]
res_sparsegpt = [9.6887,9.6158,8.0677,7.8392,7.5133,7.4339,7.2205, 7.21]

x_axis = [1,2,8,16,32,64,128,256]
plt.figure(figsize=(7,4.5))
plt.plot(res_sparsegpt, linewidth=3.0, marker="v", markersize=10, alpha=0.9, color="blue", label="SparseGPT")
plt.plot(res_simple_metric, linewidth=3.0, marker="^", markersize=10, alpha=0.9, color="red", label="Wanda")
plt.xticks([0,1,2,3,4,5,6,7],x_axis)
plt.yticks([7.5, 8.0, 8.5, 9.0, 9.5],[7.5, 8.0, 8.5, 9.0, 9.5])
plt.legend(fontsize=legendsize)
plt.xlabel("\# Calibration Samples",fontsize=labelsize)
plt.ylabel("Perplexity",fontsize=labelsize)
plt.title("LLaMA-7B",fontsize=titlesize)
plt.grid(True, linestyle='--')
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.savefig("fig.pdf", dpi=200, bbox_inches='tight', transparent=False)

red = "#FF8988"
orange = "#FECC81"
blue = "#6098FF"
green = "#77B25D"
purple = "#B28CFF"
# color palette
color_d = [red, orange, blue, green, purple]
# show all colors in a grid of squares
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
for i, color in enumerate(color_d):
ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color))
ax.set_xlim(0, len(color_d))
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
plt.show()
