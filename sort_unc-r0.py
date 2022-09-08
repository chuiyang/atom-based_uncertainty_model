import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

load_remain_data = pd.read_csv('./saved_models/qm9_130k_rbf_unscale_atl_rand_bs50/active_iter0/saved_data/remain_pred.csv').values
np.random.seed(10)
np.random.shuffle(load_remain_data)
train_data = pd.read_csv('./saved_models/qm9_130k_rbf_unscale_atl_rand_bs50/active_iter0/saved_data/train_full.csv').values
val_data = pd.read_csv('./saved_models/qm9_130k_rbf_unscale_atl_rand_bs50/active_iter0/saved_data/val_full.csv').values


k_samples = load_remain_data.shape[0] // 6
print(f'amount of remain data at first round: {load_remain_data.shape[0]}')
print(f'arg.k_samples: {k_samples}')
kremain_data = load_remain_data[:k_samples, :2]
load_remain_data = load_remain_data[k_samples:, :2]
concat_train_val_kremain = np.vstack((train_data, val_data, kremain_data))


np.random.shuffle(concat_train_val_kremain)

new_amount_of_train_data = int(concat_train_val_kremain.shape[0]*0.9)

train_data = concat_train_val_kremain[:new_amount_of_train_data, :]
val_data = concat_train_val_kremain[new_amount_of_train_data:, :]

print(f'train_data: {train_data.shape}, val_data: {val_data.shape}, remain: {load_remain_data.shape}')


for dataset, name in [(train_data, 'train'), (val_data, 'val'), (load_remain_data, 'remain')]:
    pd.DataFrame(dataset, columns=['smiles', 'true']).to_csv(os.path.join('./saved_models/qm9_130k_rbf_unscale_atl_rand_bs50/active_iter1/saved_data', name + '_full.csv'), index=False)
