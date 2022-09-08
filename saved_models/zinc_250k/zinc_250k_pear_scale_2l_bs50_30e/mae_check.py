import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

maes = []
rmses = []
for i in range(30):
    file = pd.read_csv(f'./each_pred/zinc_250k_test_pear_scale_2l_bs50_30e_{i}.csv').values
    true = file[:, 1]
    pred = file[:, 2]
    mae = np.mean(abs(true-pred))
    rmse = np.sqrt(np.mean(abs(true-pred)**2))
    print(f'model {i}: RMSE = {rmse:.6f} MAE = {mae:.6f}')
    maes.append(mae)
    rmses.append(rmse)


file = pd.read_csv(f'./fold_0/zinc_250k_test_pear_scale_2l_bs50_30e.csv').values
true = file[:, 1]
pred = file[:, 2]
mae = np.mean(abs(true-pred))
rmse = np.sqrt(np.mean(abs(true-pred)**2))
print(f'ensemble model : RMSE = {rmse:.6f} MAE = {mae:.6f}')

fig, ax = plt.subplots(figsize=(2, 4))
plt.scatter(np.zeros(30), maes, c='C0', alpha=0.7)
# plt.errorbar(1, np.mean(maes), yerr=np.std(maes), capsize=3, label='Single Model', marker='o') 
plt.scatter(0, mae, c='red', label='Ensemble Model', alpha=0.7)
plt.legend(bbox_to_anchor=(-0.12,-0.1,1.2,0.5), loc="lower left", mode="expand", borderaxespad=0, fontsize=7)
plt.ylabel('MAE')
ax.get_xaxis().set_visible(False)
plt.title('Zinc15')
plt.tight_layout()

plt.savefig('./mae_check.png', dpi=300)