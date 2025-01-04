import json

dataset_name = 'Books'
user_prompt = f'data/data_sort_by_time/{dataset_name}/user_prompt_{dataset_name}.json'
save_path = f'data/data_sort_by_time/{dataset_name}/rec_item_{dataset_name}.json'

with open(user_prompt, 'r') as f:
    user_prompt = json.load(f)

rec_item = {}
for item in user_prompt.items():
    uid = item[0]
    listx = item[1]
    for l in reversed(listx):
        if l['overall'] >= 3:
            rec_item[uid] = l['asin']

with open(save_path, 'w') as f:
    json.dump(rec_item, f, indent=4)
