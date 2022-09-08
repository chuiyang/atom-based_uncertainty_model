import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file1_ale = pd.read_csv('./qm9_130k_pear_scale_2l_30e_lr/fold_0/qm9_130k_smiles0_test_pear_scale_2l_30e_lr.csv').values[:, 3]
file2_ale = pd.read_csv('./qm9_130k_pear_scale_2l_30e_lr_post/fold_0/qm9_130k_smiles0_test_pear_scale_2l_30e_lr_post.csv').values[:, 3]

normal = 0
abnormal = 0
total = 0 


for ale1, ale2 in zip(file1_ale, file2_ale):
    if ale1 - ale2 > 0:  # tune to lower
        normal += 1
        # if (abs(ale1 - ale2) > ale1/2) and (ale1 > 10):
        #     print(ale1, ale2)
    else:
        abnormal += 1
        # print(abs(ale1 - ale2))

    total += 1

print(total, normal, abnormal)
print(len(file2_ale))

n = 13100
# p1 = sorted(file1_ale)[n:]
# p2 = sorted(file2_ale)[n:]
# print(p1[0], p2[0])
p1 = sorted(file1_ale)[:n]
p2 = sorted(file2_ale)[:n]
print(p1[-1], p2[-1])

fig, ax = plt.subplots(figsize=(7, 4))
ax2=ax.twinx()
# l1 = ax.plot(np.arange(n, len(file1_ale)), p1, c='C0', label='ale(before)')
# l2 = ax2.plot(np.arange(n, len(file2_ale)), p2, c='C1', label='ale(after)')
l1 = ax.plot(np.arange(0, n), p1, c='C0', label='ale(before)')
l2 = ax2.plot(np.arange(0, n), p2, c='C1', label='ale(after)')
lns = l1 + l2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs)
ax.set_ylabel('ale_unc (before recali)', c='C0')
ax2.set_ylabel('ale_unc (before recali)', c='C1')
# ax2.legend(loc=2)
plt.savefig('./ale_compare.png', dpi=400)