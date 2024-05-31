from pathlib import Path
import regex as re
import json
import datasets as dts
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, BitsAndBytesConfig, pipeline, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import LoraConfig, PeftModel, PeftConfig, prepare_model_for_kbit_training, get_peft_model
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

access_token = 'YOUR_HF_TOKEN_HERE'

model_id = 'mistralai/Mistral-7B-Instruct-v0.2'

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, token=access_token)

order = 'tqa'

question = 'Катеогорія тексту: політика, технології, бізнес, спорт, чи новини?'

def data_to_prompt(text, target, question=question, category: bool = True, order=order):
    if question:
        if order == 'qta':
            prompt = f'{question}\nТекст: {text}'
        elif order == 'tqa':
            prompt = f'Текст: {text}\n{question}'
    else:
        prompt =  f'Текст: {text}'
    if category:
        prompt = prompt + f'\nКатеогорія:'
    return prompt


def generate_examples(df: pd.DataFrame, tokenizer, n_examples_per_label: int = 2, max_text_len: int = 200,
                      question: str = None, random_state: int = None):
    prompt_data = df.groupby('target_name', group_keys=False)\
        .apply(lambda x: x.sample(n_examples_per_label, random_state=random_state))

    enc = tokenizer(prompt_data['text'].tolist(), add_special_tokens=False)

    enc_short = [e[:max_text_len] for e in enc.data['input_ids']]

    short_texts = tokenizer.batch_decode(enc_short)

    prompt_data['short_text'] = short_texts

    prompt_data['short_text'] = prompt_data['short_text'] + '...'

    prompt_parts = prompt_data.sample(frac=1, random_state=random_state)\
        .apply(lambda x: data_to_prompt(x['short_text'],
                                        x['target_name'],
                                        question=question,
                                        category=True),
               axis=1)

    prompt = prompt_parts.str.cat(sep='\n')

    return prompt


def create_prompt_dict(examples_df: pd.DataFrame, task_text: str, tokenizer, n_examples_per_label: int = 2, max_text_len: int = 200,
                       question: str = None, random_state: int = None, order: str = order, category: bool = False) -> dict:

    prompt_data = examples_df[examples_df['text'] != task_text].groupby('target', group_keys=False)\
        .apply(lambda x: x.sample(n_examples_per_label, random_state=random_state))\
        .reset_index(drop=True)

    # print(len(prompt_data))

    enc = tokenizer(prompt_data['text'].tolist(), add_special_tokens=False)

    enc_short = [e[:max_text_len] for e in enc.data['input_ids']]

    short_texts = tokenizer.batch_decode(enc_short)

    prompt_data['short_text'] = short_texts

    prompt_data['short_text'] = prompt_data['short_text'] + '...'

    prompt = []
    for i, row in prompt_data.sample(frac=1.0, random_state=random_state).reset_index(drop=True).iterrows():
        user_text = data_to_prompt(row['short_text'], None, question=question, category=category, order=order) #+ '\nКатеогорія тексту: політика, технології, бізнес, спорт, чи новини?'#'\nКатегорія: '
        assistant_text = row['target'] #' ' + row['target']
        # print(assistant_text)
        prompt.append({'role': 'user', 'content': user_text})
        prompt.append({'role': 'assistant', 'content': assistant_text})

    enc = tokenizer(task_text, add_special_tokens=False)

    enc_short = enc['input_ids'][:max_text_len]

    task_text_short = tokenizer.decode(enc_short) + '...'

    last_user_text = data_to_prompt(task_text_short, None, question=question, category=category, order=order) #+ '\nКатеогорія тексту: політика, технології, бізнес, спорт, чи новини?'# '\nКатегорія: '

    prompt.append({'role': 'user', 'content': last_user_text})

    return prompt


def create_prompt(example_df: pd.DataFrame, text: str, prompt_template: str, tokenizer, max_text_len: int = 180):
    text_row = f"Текст: {text}\nКатеогорія тексту: політика, технології, бізнес, спорт, чи новини?"
    if inst_tokens:
        text_row = "[INST] " + text_row + "[/INST]"

    text_row = text_row + ' ' + output

    if eos_tokens:
        text_row = '<s>' + text_row + '</s>'
    return text_row

ex_per_label = 3
max_text_len = 180
category = True

in_dir = Path('IN_DATA_DIRECTORY_HERE')
reports_dir = Path('OUT_DATA_DIRECTORY_HERE')

train_df = pd.read_json(in_dir / 'train.jsonl', orient='records', lines=True)
test_df = pd.read_json(in_dir.parents[0].resolve() / 'test.jsonl', orient='records', lines=True)

random_seeds = np.random.randint(1,9999,(len(test_df)))
np.save('random_seeds.pkl', random_seeds, allow_pickle=True)

prompts = []
for i, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    pr = create_prompt_dict(train_df, row['text'], tokenizer, n_examples_per_label=ex_per_label, order=order,
                            max_text_len=max_text_len, question=question, random_state=random_seeds[i], category=category)
    prompts.append(pr)


def dict_to_text_prompt(prompt: dict, inst_tokens: bool = True, eos_tokens: bool = False):
    text_prompt = ''
    for part in prompt[:-1]:
        text = part['content']
        if part['role'] == 'user':
            text += ' '
            if inst_tokens:
                text = "[INST] " + text + "[/INST]"
        text_prompt += text
        if part['role'] == 'assistant':
            text_prompt += '\n'
    if eos_tokens:
        text_prompt = '<s>' + text_prompt + '</s>'
    final_inst = prompt[-1]['content']
    final_inst += ' '
    if inst_tokens:
        final_inst = "[INST] " + final_inst + "[/INST]"
    text_prompt += final_inst
    return text_prompt

text_prompts = [dict_to_text_prompt(p, eos_tokens=True) for p in prompts]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

adapters_idx = [
    'ysapolovych/mistral7b-8s-uan-adapter',
    'ysapolovych/mistral7b-16s-uan-adapter',
    'ysapolovych/mistral7b-32s-uan-adapter',
]

adapter_id = adapters_idx[2]

model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}, quantization_config=bnb_config, token=access_token, return_dict=True)

merged_model= PeftModel.from_pretrained(model, adapter_id, adapter_name=adapter_id, token=access_token)

prompt_source_shot = in_dir.parts[-1]

run = {
    'experiment_n': 0,
    'prompt_source_shot': prompt_source_shot,
    'adapter': adapter_id,
    'demonstrations': True,
    'order': order,
    'test_data_size': 3000
}


class StoppingCriteriaSub(StoppingCriteria):
    # https://discuss.huggingface.co/t/implimentation-of-stopping-criteria-list/20040
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if tokenizer.decode(stop) == tokenizer.decode(last_token):
                return True
        return False

stop_words = ["\n", "Текст"]
stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

generation_kwargs = {
    "max_length": 2560,  # Maximum length of the generated text
    "max_new_tokens": 20,  # Maximum number of new tokens to generate
}

generator = pipeline(task="text-generation", model=merged_model, tokenizer=tokenizer, **generation_kwargs)

batch_size = 1000
length = len(prompts)
j = 0

categories_regex = '|'.join(train_df['target'].unique().tolist())

for i in range(0, length, batch_size):
    results = []
    print(f'Processing documents {i}:{i+batch_size}')
    for pr in tqdm(prompts[i:i+batch_size]):
        res = generator(pr)
        results.append(res[0]['generated_text'])

    predictions = []
    for i, res in enumerate(results):
        pred = res[-1]['content']
        pred = re.search(categories_regex, pred, re.IGNORECASE)
        if pred is None:
            pred = ''
        else:
            pred = pred[0]
        predictions.append(pred.lower())

    with open(reports_dir / 'mistral_test_res.txt', 'a', encoding='utf8') as f:
        for index, p in enumerate(predictions):
            if index == 0:
                f.write(f'\n{p}\n')
            if index == len(predictions) - 1:
                f.write(f'{p}')
            else:
                f.write(f'{p}\n')

    with open(reports_dir / 'mistral_test_res_batch_{j}.txt', 'w', encoding='utf8') as f:
        for p in predictions:
            f.write(f'{p}\n')
    j += 1