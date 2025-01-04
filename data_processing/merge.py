import json
import math
import re

datasets_name = 'Books'  # TripAdvisor
path = f'data/portrait/{datasets_name}'
save_path = f'data/portrait/{datasets_name}/portrait_text/'
items = math.floor(6052 / 100) if datasets_name == 'movie_and_tv' else math.floor(8521 / 100)
users = math.floor(4483 / 100) if datasets_name == 'movie_and_tv' else math.floor(4684 / 100)

data_items = {}
data_users = {}
for n in range(0, items+1):
    temp_path = path + '/item_portrait.json_' + str(n) + '.json'
    with open(temp_path, 'r') as f:
        data = json.load(f)
        for item in data.items():
            data_items[item[0]] = item[1]
with open(save_path + 'item_portrait.json', 'w') as f:
    json.dump(data_items, f, indent=4)

for n in range(0, users+1):
    temp_path = path + '/user_portrait.json_' + str(n) + '.json'
    with open(temp_path, 'r') as f:
        data = json.load(f)
        for item in data.items():
            data_users[item[0]] = item[1]
with open(save_path + 'user_portrait.json', 'w') as f:
    json.dump(data_users, f, indent=4)
