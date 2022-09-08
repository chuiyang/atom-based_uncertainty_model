"""
because molecules with N after recalibration get larger aleatoric uncertainty.
check error after ensemble, does molecules with N have larger error after ensemble?
"""

import pandas as pd
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt


preds_array = []
aleas_array = []
for model in range(15):
    file_before = pd.read_csv(f'./qm9_130k_pear_scale_2l_15e_lr_std1/each_pred/qm9_130k_noiseO_test_pear_scale_2l_15e_lr_std1_{model}.csv')
    preds_array.append(file_before['preds_Hf'].values)
    aleas_array.append(np.sqrt(file_before['Hf_ale_unc'].values))

preds_array = np.array(preds_array)
aleas_array = np.array(aleas_array)
print(f'aleas_array: {aleas_array[:, :5]}')
print(f'preds_array.shape: {preds_array.shape}')
assert preds_array.shape[0] == 15


file_after = pd.read_csv('./qm9_130k_pear_scale_2l_15e_lr_std1_post/fold_0/qm9_130k_noiseO_test_pear_scale_2l_15e_lr_std1_post.csv')

smiles = file_after['smiles'].values
preds_ens = file_after['preds_Hf'].values
true = file_after['true_Hf'].values
error = abs(preds_ens-true)
ale_unc = np.sqrt(file_after['Hf_ale_unc'].values)
epi_unc = file_after['Hf_epi_unc'].values

numN_list = []
for smile in smiles:
    mol = Chem.MolFromSmiles(smile)
    numN = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'O':
            numN += 1
    numN = 3 if numN > 3 else numN
    numN_list.append(numN)

N3 = [0 if i == 3 else -1 for i in numN_list]

for i in range(15):

    if i == 0:
        fig, axs = plt.subplots(2, 2)

    preds_i = preds_array[i]
    aleas_i = aleas_array[i]
    abnormal_count = {0: 0, 1: 0, 2: 0, 3: 0}
    error_i = {0: 0, 1: 0, 2: 0, 3: 0}
    error_e = {0: 0, 1: 0, 2: 0, 3: 0}

    ee_minus_ei = []
    p0x, p1x, p2x, p3x, p0y, p1y, p2y, p3y = [], [], [], [], [], [], [], []

    a0, a1, a2, a3 = 0, 0, 0, 0

    for j, (nN, pi, t, pe, ai, ae) in enumerate(zip(numN_list, preds_i, true, preds_ens, aleas_i, ale_unc)):
        if abs(pi-t) < abs(pe-t):  # model_i error < model_ens error, which is abnormal
            abnormal_count[nN] += 1
            if nN == 3:
                N3[j] += 1
        error_i[nN] += abs(pi-t)
        error_e[nN] += abs(pe-t)

        ee_minus_ei.append((abs(pe-t)-abs(pi-t)))

        color = ['C0', 'C1', 'C2', 'C3']
        label = ['0N', '1N', '2N', '3N_up']
        if i == 0:
            if nN == 0:
                p0x.append(abs(pe-t)-abs(pi-t))
                p0y.append(ae-ai)
                if ((abs(pe-t)-abs(pi-t)) > 2 and (ae-ai) < 2) or ((abs(pe-t)-abs(pi-t)) < 2 and (ae-ai) > 2):
                    a0 += 1
            elif nN == 1:
                p1x.append(abs(pe-t)-abs(pi-t))
                p1y.append(ae-ai)   
                if ((abs(pe-t)-abs(pi-t)) > 2 and (ae-ai) < 2) or ((abs(pe-t)-abs(pi-t)) < 2 and (ae-ai) > 2):
                    a1 += 1
            elif nN == 2:
                p2x.append(abs(pe-t)-abs(pi-t))
                p2y.append(ae-ai)         
                if ((abs(pe-t)-abs(pi-t)) > 2 and (ae-ai) < 2) or ((abs(pe-t)-abs(pi-t)) < 2 and (ae-ai) > 2):
                    a2 += 1
            elif nN == 3:
                p3x.append(abs(pe-t)-abs(pi-t))
                p3y.append(ae-ai)
                if ((abs(pe-t)-abs(pi-t)) > 2 and (ae-ai) < 2) or ((abs(pe-t)-abs(pi-t)) < 2 and (ae-ai) > 2):
                    a3 += 1
            # plt.scatter(abs(pe-t)-abs(pi-t), ae-ai, c=color[nN])
    if i == 0:
        axs[0, 0].plot([min(p0x), max(p0x)], [0, 0], c='black', linewidth=1, alpha=0.5, linestyle='--')
        axs[0, 0].plot([0, 0], [min(p0y), max(p0y)], c='black', linewidth=1, alpha=0.5, linestyle='--')
        axs[0, 1].plot([min(p1x), max(p1x)], [0, 0], c='black', linewidth=1, alpha=0.5, linestyle='--')
        axs[0, 1].plot([0, 0], [min(p1y), max(p1y)], c='black', linewidth=1, alpha=0.5, linestyle='--')
        axs[1, 0].plot([min(p2x), max(p2x)], [0, 0], c='black', linewidth=1, alpha=0.5, linestyle='--')
        axs[1, 0].plot([0, 0], [min(p2y), max(p2y)], c='black', linewidth=1, alpha=0.5, linestyle='--')
        axs[1, 1].plot([min(p3x), max(p3x)], [0, 0], c='black', linewidth=1, alpha=0.5, linestyle='--')
        axs[1, 1].plot([0, 0], [min(p3y), max(p3y)], c='black', linewidth=1, alpha=0.5, linestyle='--')
        axs[0, 0].scatter(p0x, p0y, c=color[0], label='0N', s=5)
        axs[0, 1].scatter(p1x, p1y, c=color[1], label='1N', s=5)
        axs[1, 0].scatter(p2x, p2y, c=color[2], label='2N', s=5)
        axs[1, 1].scatter(p3x, p3y, c=color[3], label='3N_up', s=5)

        axs[0, 0].legend()
        axs[0, 1].legend()
        axs[1, 0].legend()
        axs[1, 1].legend()
        axs[0, 0].set_title(f'{a0}/{len(p0x)}')
        axs[0, 1].set_title(f'{a1}/{len(p1x)}')
        axs[1, 0].set_title(f'{a2}/{len(p2x)}')
        axs[1, 1].set_title(f'{a3}/{len(p3x)}')
        # axs[0, 0].set_xlim([-5, 5])
        # axs[0, 0].set_ylim([-5, 5])        
        # axs[0, 1].set_xlim([-5, 5])
        # axs[0, 1].set_ylim([-5, 5])        
        # axs[1, 0].set_xlim([-5, 5])
        # axs[1, 0].set_ylim([-5, 5])        
        # axs[1, 1].set_xlim([-5, 5])
        # axs[1, 1].set_ylim([-5, 5])
        axs[0, 0].set_xlabel('Error(ens)-Error(i)')
        axs[0, 0].set_ylabel('Alea(post)-Alea(i)') 
        axs[0, 1].set_xlabel('Error(ens)-Error(i)')
        axs[0, 1].set_ylabel('Alea(post)-Alea(i)') 
        axs[1, 0].set_xlabel('Error(ens)-Error(i)')
        axs[1, 0].set_ylabel('Alea(post)-Alea(i)') 
        axs[1, 1].set_xlabel('Error(ens)-Error(i)')
        axs[1, 1].set_ylabel('Alea(post)-Alea(i)')   
        fig.tight_layout()

        # axs[0, 0].set_xlim(left=0)
        # axs[0, 1].set_xlim(left=0)
        # axs[1, 0].set_xlim(left=0)
        # axs[1, 1].set_xlim(left=0)        
        # axs[0, 0].set_ylim(bottom=0)
        # axs[0, 1].set_ylim(bottom=0)
        # axs[1, 0].set_ylim(bottom=0)
        # axs[1, 1].set_ylim(bottom=0)

    # plt.scatter(numN_list, ee_minus_ei, s=2)
    # plt.plot([-1, 4], [0, 0], c='black')
    # plt.xticks(np.arange(0, 4))    
    # plt.ylim([-1, max(ee_minus_ei)+1])
    if i == 0:
        plt.savefig(f'./model_{i}.png', dpi=300)

    

    
        
    
    print(f'model: {i}')
    for key, val in abnormal_count.items():
        numN_ = numN_list.count(key)
        print(f'N_{key} ({numN_}): {val} ({(val/numN_*100):.2f}%)| MAE_e: {(error_e[key]/numN_):.3f}, MAE_i: {(error_i[key]/numN_):.3f}')
    print(f'total_MAE_e: {np.mean(abs(preds_ens-true)):.3f}, total MAE_i: {np.mean(abs(preds_i-true)):.3f}')
    print(f'\n')

    

    

# check pred
print(f'preds_ens[:5]: {[i for i in preds_ens[10:15]]}')
print(f'preds_array[:5]: {[i for i in np.mean(preds_array, axis=0)[10:15]]}')

sumN3 = 0
for j in range(15, -1, -1):
    print(f'{j}: {N3.count(j)}')
    sumN3 += N3.count(j)
print(sumN3)