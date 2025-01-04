import json

dataset_name = 'Books'

item_list_path = f'data/data_gcn/{dataset_name}/item_list.txt'
user_list_path = f'data/data_gcn/{dataset_name}/user_list.txt'

item_raw_id2embed_path = f'data/portrait/{dataset_name}/portrait_embedding/item_portrait_embedding.json'
user_raw_id2embed_path = f'data/portrait/{dataset_name}/portrait_embedding/user_portrait_embedding.json'

item_id2embed_path = f'data/data_gcn/{dataset_name}/item_id2embed.json'
user_id2embed_path = f'data/data_gcn/{dataset_name}/user_id2embed.json'

itemid_dict = {}
with open(item_list_path, 'r') as f:
    lines = f.readlines()
for line in lines[1:]:
    list = line.split(' ')
    itemid_dict[list[0]] = list[1].rstrip()
item_id2embed_dict = {}
with open(item_raw_id2embed_path, 'r') as f:
    data = json.load(f)
for item in data.items():
    temp = {}
    raw_id = item[0]
    embed = item[1]
    id = itemid_dict[raw_id]
    item_id2embed_dict[id] = embed
with open(item_id2embed_path, 'w') as f:
    json.dump(item_id2embed_dict, f, indent=4)

userid_dict = {}
with open(user_list_path, 'r') as f:
    lines = f.readlines()
for line in lines[1:]:
    list = line.split(' ')
    userid_dict[list[0]] = list[1].rstrip()
user_id2embed_dict = {}
with open(user_raw_id2embed_path, 'r') as f:
    data = json.load(f)
for item in data.items():
    temp = {}
    raw_id = item[0]
    embed = item[1]
    id = userid_dict[raw_id]
    user_id2embed_dict[id] = embed
with open(user_id2embed_path, 'w') as f:
    json.dump(user_id2embed_dict, f, indent=4)

item_portrait_path = f'data/data_gcn/{dataset_name}/item_id2embed.json'
user_portrait_path = f'data/data_gcn/{dataset_name}/user_id2embed.json'

item_save_path = f'data/data_gcn/{dataset_name}/item_embed_list.json'
user_save_path = f'data/data_gcn/{dataset_name}/user_embed_list.json'

with open(item_portrait_path, 'r') as f:
    data = json.load(f)
len_dict = len(data)
print(len_dict)
item_list = [0] * len_dict
for item in data.items():
    id = int(item[0])
    embed = item[1]
    item_list[id] = embed
with open(item_save_path, 'w') as f:
    json.dump(item_list, f)

with open(user_portrait_path, 'r') as f:
    data = json.load(f)
len_dict = len(data)
print(len_dict)
user_list = [0] * len_dict
for item in data.items():
    id = int(item[0])
    embed = item[1]
    user_list[id] = embed
with open(user_save_path, 'w') as f:
    json.dump(user_list, f)
