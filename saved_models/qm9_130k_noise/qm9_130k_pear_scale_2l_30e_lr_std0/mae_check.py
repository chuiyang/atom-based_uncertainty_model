import pandas as pd
import numpy as np

for i in range(15):
    file = pd.read_csv(f'./each_pred/qm9_130k_noise_test_pear_scale_2l_30e_lr_std0_{i}.csv').values
    true = file[:, 1]
    pred = file[:, 2]
    mae = np.mean(abs(true-pred))
    rmse = np.sqrt(np.mean(abs(true-pred)**2))
    print(f'model {i}: RMSE = {rmse:.6f} MAE = {mae:.6f}')