import json

item_list_path = 'data/video_games/item_list.txt'
user_list_path = 'data/video_games/user_list.txt'

item_raw_id2embed_path = 'data/video_games/item_raw_id2embed.json'
user_raw_id2embed_path = 'data/video_games/user_raw_id2embed.json'

item_id2embed_path = 'data/video_games/item_id2embed.json'
user_id2embed_path = 'data/video_games/user_id2embed.json'

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
    item_id2embed_dict[id]=embed
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
    user_id2embed_dict[id]=embed
with open(user_id2embed_path, 'w') as f:
    json.dump(user_id2embed_dict, f, indent=4)
