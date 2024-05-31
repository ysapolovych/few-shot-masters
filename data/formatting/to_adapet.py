#%%
import json
from pathlib import Path
import pandas as pd

cwd = Path(__file__).parents[1].resolve()
in_path = cwd / 'data' / 'original'
out_path = cwd / 'data' / 'adapet'
out_8shot = out_path / '8shot'
out_16shot = out_path / '16shot'

out_8shot.mkdir(exist_ok=True, parents=True)
out_16shot.mkdir(exist_ok=True, parents=True)

random_seed = 99

with open(cwd / 'data' / 'label_to_id.json', 'r') as f:
    label_to_id = json.load(f)

train = pd.read_csv(in_path / 'train.csv')
test = pd.read_csv(in_path / 'test.csv')

# %%
train['LBL'] = train['target'].map(label_to_id)
test['LBL'] = test['target'].map(label_to_id)

#%%
train_indexi = []
for n_shot, out_dir in {8: out_8shot, 16: out_16shot}.items():
    few_shot_train = train.groupby('target', group_keys=False)\
        .apply(lambda x: x.sample(n_shot, random_state=random_seed))
    train_indexi.append(few_shot_train.index)
    few_shot_train['LBL'] = few_shot_train['LBL'].astype('str')
    few_shot_train.rename(columns={'text': 'TEXT1'})\
        [['TEXT1', 'LBL']]\
        .to_json(out_dir / 'train.jsonl', orient='records', lines=True, force_ascii=False)
#%%
i = 0
for n_shot, out_dir in {8: out_8shot, 16: out_16shot}.items():
    few_shot_val = train.drop(index=train_indexi[i])\
        .groupby('target', group_keys=False)\
        .apply(lambda x: x.sample(n_shot, random_state=random_seed))
    train_indexi.append(few_shot_val.index)
    few_shot_val['LBL'] = few_shot_val['LBL'].astype('str')
    few_shot_val.rename(columns={'text': 'TEXT1'})\
        [['TEXT1', 'LBL']]\
        .to_json(out_dir / 'val.jsonl', orient='records', lines=True, force_ascii=False)
    i += 1
#%%
test['LBL'] = test['LBL'].astype('str')
test.rename(columns={'text': 'TEXT1'})\
    [['TEXT1', 'LBL']]\
    .to_json(out_path / 'test.jsonl', orient='records', lines=True, force_ascii=False)
# %%
test.rename(columns={'text': 'TEXT1'})\
    .sample(300, random_state=random_seed)\
    [['TEXT1', 'LBL']]\
    .to_json(out_path / 'very_short_test.jsonl', orient='records', lines=True, force_ascii=False)
#%%
test.rename(columns={'text': 'TEXT1'})\
    .sample(1000, random_state=random_seed)\
    [['TEXT1', 'LBL']]\
    .to_json(out_path / 'short_test.jsonl', orient='records', lines=True, force_ascii=False)