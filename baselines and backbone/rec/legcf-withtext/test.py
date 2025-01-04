import json
import random

item_list_path = 'data/video_games/item_list.txt'
user_list_path = 'data/video_games/user_list.txt'

item_save_path = 'data/video_games/item_embed_list.json'
user_save_path = 'data/video_games/user_embed_list.json'

size = 128

item_list = []
with open(item_list_path) as f:
    lines = f.readlines()
for line in lines[1:]:
    item_list.append([random.uniform(0, 1) for _ in range(size)])
user_list = []
with open(user_list_path) as f:
    lines = f.readlines()
for line in lines[1:]:
    user_list.append([random.uniform(0, 1) for _ in range(size)])

with open(item_save_path, 'w') as f:
    json.dump(item_list, f)
with open(user_save_path, 'w') as f:
    json.dump(user_list, f)
