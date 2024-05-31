#%%
from pathlib import Path
import pandas as pd

cwd = Path(__file__).parents[1].resolve()
in_path = cwd / 'data' / 'original'
out_path = cwd / 'data' / 'setfit'
out_8shot = out_path / '8shot'
out_16shot = out_path / '16shot'

out_8shot.mkdir(exist_ok=True, parents=True)
out_16shot.mkdir(exist_ok=True, parents=True)

random_seed = 99

train = pd.read_csv(in_path / 'train.csv')
test = pd.read_csv(in_path / 'test.csv')
print(train.shape)
print(test.shape)
#%%
# train_onehot = pd.get_dummies(train, columns=['target'], dtype=int)
# test_onehot = pd.get_dummies(test, columns=['target'], dtype=int)
# # train.rename(columns={'target': 'target_name'}, inplace=True)
# train['onehot'] = train_onehot[[c for c in train_onehot.columns if 'target' in c]].values.tolist()
# test['onehot'] = test_onehot[[c for c in test_onehot.columns if 'target' in c]].values.tolist()
# #%%
# train[['target', 'onehot']].drop_duplicates(subset=['target']).sort_values('target')
# #%%
# test[['target', 'onehot']].drop_duplicates(subset=['target']).sort_values('target')

train_indexi = []
for n_shot, out_dir in {8: out_8shot, 16: out_16shot}.items():
    few_shot_train = train.groupby('target', group_keys=False)\
        .apply(lambda x: x.sample(n_shot, random_state=random_seed))
    train_indexi.append(few_shot_train.index.tolist())
    # few_shot_train = few_shot_train.rename(columns={'target': 'target_name', 'onehot': 'target'})
    few_shot_train.to_json(out_dir / 'train.jsonl', orient='records', lines=True, force_ascii=False)
# %%
i = 0
for n_shot, out_dir in {8: out_8shot, 16: out_16shot}.items():
    few_shot_val = train.drop(index=train_indexi[i])\
        .groupby('target', group_keys=False)\
        .apply(lambda x: x.sample(n_shot, random_state=random_seed))
    train_indexi[i].extend(few_shot_val.index.tolist())
    # few_shot_val = few_shot_val.rename(columns={'target': 'target_name', 'onehot': 'target'})
    few_shot_val.to_json(out_dir / 'valid.jsonl', orient='records', lines=True, force_ascii=False)
    i += 1
#%%
i = 0
for n_shot, out_dir in {8: out_8shot, 16: out_16shot}.items():
    train.drop(index=train_indexi[i])\
        .to_json(out_dir / 'unlabeled.jsonl', orient='records', lines=True, force_ascii=False)
        # .to_csv(out_dir / 'unlabeled.csv', index=False)
    i += 1
#%%
# test = test.rename(columns={'target': 'target_name', 'onehot': 'target'})
test.to_json(out_path / 'test.jsonl', orient='records', lines=True, force_ascii=False)

test.sample(frac=0.2, random_state=random_seed)\
    .to_json(out_path / 'test_short.jsonl', orient='records', lines=True, force_ascii=False)

# test.sample(2000, random_state=random_seed)\
#     .to_json(out_path / 'short_test.jsonl', orient='records', lines=True, force_ascii=False)
# %%
