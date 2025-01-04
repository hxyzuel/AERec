import json

item_portrait_path = 'data/video_games/item_id2embed.json'
user_portrait_path = 'data/video_games/user_id2embed.json'

item_save_path = 'data/video_games/item_embed_list.json'
user_save_path = 'data/video_games/user_embed_list.json'

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
