#%%
import json
from pathlib import Path
import pandas as pd

cwd = Path(__file__).parents[1].resolve()
in_path = cwd / 'temp' / 'original'
out_path = cwd / 'data' / 'perfect'
out_8shot = out_path / '8shot'
out_16shot = out_path / '16shot'

out_8shot.mkdir(exist_ok=True, parents=True)
out_16shot.mkdir(exist_ok=True, parents=True)

random_seed = 99

# with open(cwd / 'data' / 'label_to_id.json', 'r') as f:
#     label_to_id = json.load(f)

train = pd.read_csv(in_path / 'train.csv')
test = pd.read_csv(in_path / 'test.csv')
#%%
dict_verbalizer = {str(n): label for n, label in enumerate(test['target'].unique())}
dict_verbalizer_reverse = {label: n for n, label in enumerate(test['target'].unique())}
with open(out_path / 'verbalizers.json', 'w') as f:
    json.dump(dict_verbalizer, f, ensure_ascii=False)
#%%
train['target_name'] = train['target']
test['target_name'] = test['target']
#%%
train_indexi = []
for n_shot, out_dir in {8: out_8shot, 16: out_16shot}.items():
    few_shot_train = train.groupby('target', group_keys=False)\
        .apply(lambda x: x.sample(n_shot, random_state=random_seed))
    train_indexi.append(few_shot_train.index)
    few_shot_train['label'] = few_shot_train['target'].map(dict_verbalizer_reverse)
    few_shot_train['label'] = few_shot_train['label'].astype(str)
    few_shot_train.rename(columns={'text': 'source'})\
        [['label', 'source']]\
        .to_json(out_dir / f'train.json', orient='records', lines=True, force_ascii=False)
#%%
i = 0
for n_shot, out_dir in {8: out_8shot, 16: out_16shot}.items():
    few_shot_val = train.drop(index=train_indexi[i])\
        .groupby('target', group_keys=False)\
        .apply(lambda x: x.sample(n_shot, random_state=random_seed))
    train_indexi.append(few_shot_val.index)
    few_shot_val['label'] = few_shot_val['target_name'].map(dict_verbalizer_reverse)
    few_shot_val['label'] = few_shot_val['label'].astype(str)
    few_shot_val.rename(columns={'text': 'source'})\
        [['label', 'source']]\
        .to_json(out_dir / f'val.json', orient='records', lines=True, force_ascii=False)
    i += 1
#%%
test['label'] = test['target_name'].map(dict_verbalizer_reverse)
test['label'] = test['label'].astype(str)
test.rename(columns={'text': 'source'})\
    [['label', 'source']]\
    .to_json(out_path / f'test.json', orient='records', lines=True, force_ascii=False)