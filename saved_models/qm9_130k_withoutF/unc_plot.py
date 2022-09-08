from fileinput import filename
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
import scipy
import os
import matplotlib.ticker as mticker
import scipy.stats

"""
https://colorhunt.co/
"""


def confidence_based_calibration_func(data, filename, std):
    title = 'Confidence-based Calibration'
    f = open(os.path.join(f'{filename}_performance_std{std}.txt'), 'a')
    f.write(f'\n\n{title}\n')

    sub_data = data.values

    true = sub_data[:, 1][:, np.newaxis].astype(float)
    mean = sub_data[:, 2][:, np.newaxis].astype(float)
    var_ale = sub_data[:, 3][:, np.newaxis].astype(float)
    var_epi = sub_data[:, 4][:, np.newaxis].astype(float)
    var_total = var_epi + var_ale
    error = abs(true-mean)

    fig, ax = plt.subplots(figsize=(7, 7))

    for var, var_name in zip([var_epi, var_ale, var_total], ['epi', 'ale', 'total']):
        cali = []
        k_bin = np.linspace(0, 1, 20, endpoint=False)
        for j, con in enumerate(k_bin):
            count = 0
            for m, v, t in zip(mean, var, true):
                l_, u_ = scipy.stats.norm.interval(con, loc=m, scale=v**(1/2))  # con: confidence interval
                if l_ < t < u_:
                    count += 1
            cali.append(count/len(sub_data))
        plt.plot(k_bin, cali, linewidth=2)


        f.write(f'{var_name}\n')
        f.write(f'AUCE (l): {sum(abs(cali - k_bin)):.3f}\n')
        f.write(f'MCE (l): {(max(abs(cali - k_bin))):.3f} | ')
        f.write(f'ECE (l): {((1 / 20) * sum(abs(cali - k_bin))):.3f}\n')

        print(f'{var_name} Calibration Error Curve(AUCE, l): {sum(abs(cali - k_bin)):.3f}\n')
        print(f'Maximum Calibration Error(MCE, l): {(max(abs(cali - k_bin))):.3f}\n')
        print(f'Expected Calibration Error(ECE, l): {((1 / 20) * sum(abs(cali - k_bin))):.3f}\n')

    plt.legend(['epi', 'ale', 'total'], loc='upper left', fontsize=17)  # 'lower right'
    plt.plot([0, 1], [0, 1], '--', c='gray')
    plt.title(f'({len(sub_data)})\n MAE: {np.mean(error):.2f}, RMSE: {np.sqrt(np.mean((error**2))):.2f}', fontsize=16)

    plt.grid()
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.xlabel('Confidence Interval', fontsize=18)
    plt.ylabel('% of molecules fall in CI', fontsize=18)
    ax.set_aspect('equal', adjustable='box')


    plt.suptitle(f'{title}, {filename}', fontsize=18)
    plt.savefig(f'{filename}_cali_std{std}.png', dpi=800)
    f.close()
    plt.close()



def locateCount(count_list, unc):
    if 0 <= unc < 2:
        count_list[0] += 1
    elif 2 <= unc < 3:
        count_list[1] += 1
    elif 3 <=  unc < 4:
        count_list[2] += 1
    elif 4 <= unc < 5:
        count_list[3] += 1
    else:
        count_list[4] += 1
    return count_list

# def locateCount(count_list, unc):
#     if 0 <= unc < 5:
#         count_list[0] += 1
#     elif 5 <= unc < 10:
#         count_list[1] += 1
#     elif 10 <=  unc < 15:
#         count_list[2] += 1
#     elif 15 <= unc < 20:
#         count_list[3] += 1
#     else:
#         count_list[4] += 1
#     return count_list

# def locateCount(count_list, unc):
#     if 0 <= unc < 1:
#         count_list[0] += 1
#     elif 1 <= unc < 2:
#         count_list[1] += 1
#     elif 2 <=  unc < 3:
#         count_list[2] += 1
#     elif 3 <= unc < 4:
#         count_list[3] += 1
#     else:
#         count_list[4] += 1
#     return count_list

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{(height*100):.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

def barChart(smiles, uncertainty, x_label, unc_tag, model):

    # 0~5, 5~10, 10~15, 15~20, 20Up
    x_ticks = ['(0, 2]', '(2, 4]', '(4, 6]', '(6, 8]', '(8, inf]']
    # x_ticks = ['(0, 1]', '(1, 2]', '(2, 3]', '(3, 4]', '(4, inf]']
    y_0F = np.zeros(5) 
    y_1F = np.zeros(5) 
    y_2F = np.zeros(5) 
    y_3Fup = np.zeros(5) 
    F_dict = {0: y_0F, 1: y_1F, 2: y_2F, 3: y_3Fup}

    # locate smiles(numN) and uncertainty
    for smile, unc in zip(smiles, uncertainty):
        mol = Chem.MolFromSmiles(smile)
        numF = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'F':
                numF += 1
        
        numF = 3 if numF > 3 else numF
        F_dict[numF] = locateCount(F_dict[numF], unc)

    for i in range(4):
        classSum = np.sum(F_dict[i])
        print(f'{i}: {classSum}')
        for j in range(5):
            F_dict[i][j] = F_dict[i][j] / classSum


    # start to plot
    x = np.arange(len(x_ticks))  # the label locations
    width = 0.21  # the width of the bars
    fig, ax = plt.subplots()
    if unc_tag == 'ale':
        bar1 = ax.bar(x - 3*width/2, y_0F, width, label='0F', color='#DACC96')
        bar2 = ax.bar(x - width/2, y_1F, width, label='1F', color='#BF8B67')    
        bar3 = ax.bar(x + width/2, y_2F, width, label='2F', color='#9D5353')
        bar4 = ax.bar(x + 3*width/2, y_3Fup, width, label='3F_up', color='#632626')
    elif unc_tag == 'epi':
        bar1 = ax.bar(x - 3*width/2, y_0F, width, label='0F', color='#E7E0C9')
        bar2 = ax.bar(x - width/2, y_1F, width, label='1F', color='#C1CfC0')    
        bar3 = ax.bar(x + width/2, y_2F, width, label='2F', color='#6B7AA1')
        bar4 = ax.bar(x + 3*width/2, y_3Fup, width, label='3F_up', color='#11324D')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage (%)', fontsize=13)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_xticks(x)
    ax.set_ylim([0, 1.15])
    ax.set_xticklabels(x_ticks, fontsize=12)
    ax.yaxis.set_major_locator(mticker.FixedLocator(ax.get_yticks()))
    ax.set_yticklabels([f'{(x*100):.0f}' for x in ax.get_yticks()])
    ax.legend(bbox_to_anchor=(0.02,0.915,0.96,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4, fontsize=12)



    autolabel(bar1, ax)
    autolabel(bar2, ax)
    autolabel(bar3, ax)
    autolabel(bar4, ax)

    fig.tight_layout()

    plt.savefig(f'./F_noise_plot_{unc_tag}_14e_post-2-2468.png', dpi=650)





def uncertaintyCountNumN(filepath, unc_tag, model=None):
    file = pd.read_csv(filepath)
    smiles = file['smiles'].values
    error = abs(file['true_Hf'].values-file['preds_Hf'].values)
    ale_unc = file['Hf_ale_unc'].values
    epi_unc = file['Hf_epi_unc'].values
    unc = ale_unc if unc_tag == 'ale' else epi_unc
    x_label = 'Aleatoric Uncertainty' if unc_tag == 'ale' else 'Epistemic Uncertainty'

    # file2 = pd.read_csv(f'./qm9_130k_pear_scale_2l_15e_withoutN/fold_0/qm9_130k_withoutN_test_pear_scale_2l_15e_withoutN.csv')
    # smiles = np.hstack((smiles, file2['smiles'].values))
    # ale_unc = np.hstack((ale_unc, file2['Hf_ale_unc'].values))
    # epi_unc = np.hstack((epi_unc, file2['Hf_epi_unc'].values))
    # unc = ale_unc if unc_tag == 'ale' else epi_unc
    # title = 'Aleatoric Uncertainty' if unc_tag == 'ale' else 'Epistemic Uncertainty'
    barChart(smiles, unc, x_label, unc_tag, model)
    # confidence_based_calibration_func(file, filepath.split('/')[1], std)


if __name__ == '__main__':
    # for std in np.arange(0, 6):
    # std = 0
    # for std in [0, 1]:
    # for model in [3]:
    # for model in range(13, 15):
    #     filename = f'./qm9_130k_pear_scale_2l_15e_withoutF_seed20_smiles0-cpuall2/each_pred/qm9_130k_withoutF_test_pear_scale_2l_15e_withoutF_seed20_smiles0-cpuall2_{model}.csv'
    #     uncertaintyCountNumN(filename, 'ale', model)



    filename = f'./qm9_130k_pear_scale_2l_14e_withoutF_seed20_smiles0_post-cpuall2/fold_0/qm9_130k_withoutF_test_pear_scale_2l_14e_withoutF_seed20_smiles0_post-cpuall2.csv'
    uncertaintyCountNumN(filename, 'epi')
    uncertaintyCountNumN(filename, 'ale')
    


    
