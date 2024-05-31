"""Generate three subsets with length 1000 for inference speed tests
"""
from pathlib import Path
import numpy as np


in_path = Path('path-to-test-file')

with open(in_path / 'test.jsonl', 'r') as f:
    texts = f.read().splitlines()


for i, j in enumerate([99, 999, 42]):
    np.random.seed(j)
    idx = np.random.randint(0, 30104, 1000)
    lines = [texts[id] for id in idx]
    with open(in_path / f'sample{i}.json', 'w') as f:
        for l in lines:
            f.write(f'{l}\n')