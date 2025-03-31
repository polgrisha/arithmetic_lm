import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import torch
import pandas as pd
import numpy as np

data = torch.load('/mnt/qb/work/eickhoff/esx208/arithmetic-lm/data/patching_results_original_dataset_pythia_12b/patching_effect_residual_stream.pkl').numpy()

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Create a dataframe from the matrix
idx = ['(1) ICE 1st operand', '(+) ICE 1st operator', '(3) ICE 2nd operand', '(+) ICE 2nd operator',
       '(4) ICE 3rd operand', '(=) ICE Equal Sign', '(8) ICE Result', '(2) Task 1st operand', 
       '(+) Task 1st operator', '(2) Task 2nd operand', '(+) Task 2nd operator', 
       '(6) Task 3rd operand', '(=) Task Equal Sign']

df = pd.DataFrame(data, index=idx, columns=np.arange(36))

vmax=None

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.1)
sns.set_style("whitegrid", {'axes.grid' : False})
ax = sns.heatmap(df, cmap="Blues", annot=False, fmt=".1f", cbar=True, vmax=vmax)
# set x-axis ticks to the range of unique layers
ax.set_xticks(range(36))

# # set x-axis labels to the range of unique layers
ax.set_xticklabels(range(36), fontsize=14, rotation=0)
plt.xlabel('Layer', fontsize=16, fontname='DeJavu Serif')

for index, label in enumerate(ax.xaxis.get_ticklabels()):
    if index % 5 != 0:
        label.set_visible(False)

# Y-axis adjustments: Make labels horizontal and remove the y-axis title
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, rotation=0)
plt.ylabel('')# Remove y-axis title

# Title and other adjustments
plt.title(f'Patching Effect of the Residual Stream ', fontsize=16, fontname='DeJavu Serif')

# Adjust color bar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)

# Remove chart junk
sns.despine()

plt.savefig("/mnt/qb/work/eickhoff/esx208/arithmetic-lm/data/patching_results_original_dataset_pythia_12b/patching_effect_residual_stream.png", format='png', bbox_inches='tight')
plt.show()