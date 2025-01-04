import json
from LLM import *
import csv


def prompt_construct(item_type, cap_item_type, aspects, user_portrait, item_portrait):
    prompt = f"""
    You need to recommend the {item_type} <g> to the user <u>. \n\
    Your response should be based on the descriptions provided for both user <u> and {item_type} <g>. \n\n\
    e.g.{aspects} etc.\n\
    Ensure:\n\
    1.Format the response as:'>>Aspect_k:The reasons why recommend in this aspect'.And response should NOT include anything else.(Aspect_k stands the k-th aspect)\n\
    2.If information of Aspect_M WAS NOT in the reviews, DO NOT response:'>>Aspect_m:your inferences based on Aspect_m'.\n\
    3.Make reasonable inferences from the existing descriptions.\n\
    4.Ensure inferences are logical, clear, and reasons are specific.\n\
    5.The response must be concise, within 120 words.\n\n\
    User <u> description:\n\
    {user_portrait}\n\
    {cap_item_type} <g> description:\n\
    {item_portrait}
    """
    return prompt


dataset_name = 'movie_and_tv'  # movie_and_tv
rec_item_path = f'data/data_sort_by_time/{dataset_name}/rec_item_{dataset_name}.json'
user_portrait_path = f'data/portrait/{dataset_name}/portrait_text/user_portrait.json'
item_portrait_path = f'data/portrait/{dataset_name}/portrait_text/item_portrait.json'
aspects_path = f'data/aspects/{dataset_name}/aspects_full.txt'

save_path = f'data/data_for_fine-tune/{dataset_name}/temp.csv'

item_type = 'hotel' if dataset_name == 'TripAdvisor' else 'movie or tv'
cap_item_type = 'Hotel' if dataset_name == 'TripAdvisor' else 'Movie or tv'

aspects = ''
with open(aspects_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        aspects += line.rstrip()
        aspects += ','
with open(rec_item_path, 'r') as f:
    rec_items = json.load(f)
with open(user_portrait_path, 'r') as f:
    user_portraits = json.load(f)
with open(item_portrait_path, 'r') as f:
    item_portraits = json.load(f)

start = 800
num = 40
model = GPT3(0.9)
results = []
uids = list(rec_items.keys())[start:start + num]
iids = list(rec_items.values())[start:start + num]
for uid, iid in zip(uids, iids):
    result = {}
    if num == 0:
        break
    user_portrait = user_portraits[uid]
    item_portrait = item_portraits[iid]

    prompt = prompt_construct(item_type, cap_item_type, aspects, user_portrait, item_portrait)
    input = {}
    input['role'] = 'user'
    input['content'] = prompt

    exp = model.get_completion(prompt=[input])

    result['uid'] = uid
    result['iid'] = iid
    result['exp'] = exp
    results.append(result)

fields = ['uid', 'iid', 'exp']
with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)
