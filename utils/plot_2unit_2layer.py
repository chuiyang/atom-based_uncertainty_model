import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

scaling = ['scale', 'unscale']
pairing = ['rbf', 'cos']
output_layer = ['2unit', '2layer']

mol_filename = [f'qm9_130k_mol_{scaling_i}' for scaling_i in scaling]
atom_filename = [f'qm9_130k_{pairing_i}_{scaling_i}_{output_layer_i}' for output_layer_i in output_layer for scaling_i in scaling for pairing_i in pairing]

mol_note = [f'm_{scaling_i}' for scaling_i in scaling]
atom_note = [f'a_{pairing_i}_{scaling_i}_{output_layer_i}' for output_layer_i in output_layer for scaling_i in scaling for pairing_i in pairing]

filename_list = mol_filename + atom_filename
notes = mol_note + atom_note
mean_rmse = []
mean_mae = []

plt.figure(figsize=(17, 11))

for note, filename in zip(notes, filename_list):
    fold_log = pd.read_csv(os.path.join('./saved_models', filename, 'fold_log.csv'), header=None).values
    fold = fold_log[:, 0]
    loss = fold_log[:, 1]
    rmse = fold_log[:, 2]
    mae = fold_log[:, 3]
    mean_rmse.append(np.round(np.mean(rmse), 3))
    mean_mae.append(np.round(np.mean(mae), 3))
    print(f'{note}, {np.round(np.mean(loss), 3)}, {np.round(np.mean(rmse), 3)}, {np.round(np.mean(mae), 3)}')
    

ax1 = plt.subplot(2, 1, 1)
ax1.bar(np.arange(len(mean_rmse)), mean_rmse)
ax1.set_xticks(np.arange(len(mean_rmse)))
ax1.set_xticklabels(notes)
ax1.set_title('rmse')
ax2 = plt.subplot(2, 1, 2)
ax2.bar(np.arange(len(mean_mae)),  mean_mae)
ax2.set_xticks(np.arange(len(mean_mae)))
ax2.set_xticklabels(notes)
ax2.set_title('mae')
plt.tight_layout()
plt.savefig('./plot_2unit_2layer.png', dpi=800)




