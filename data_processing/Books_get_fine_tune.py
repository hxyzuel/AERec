import json
import random

dataset_name = 'Books'
n = 800
rec_item_path = f'data/data_sort_by_time/{dataset_name}/rec_item_{dataset_name}.json'
save_path = f'data/data_for_fine-tune/{dataset_name}/fine-tune.json'

with open(rec_item_path, 'r') as f:
    rec_items = json.load(f)

keys = list(rec_items.keys())

selected_keys = random.sample(keys, n)

result = {}

for key in selected_keys:
    result[key] = rec_items[key]

with open(save_path, 'w') as f:
    json.dump(result, f, indent=4)
