import json
import re
import voyageai
import os


class voyage(object):
    def __init__(self):
        super().__init__()
        self.client = voyageai.Client(api_key='')

    def get_embedding(self, text, model):
        # model:voyage-large-2-instruct,voyage-large-2,voyage-2
        result = self.client.embed([text], model=model)
        return result.embeddings


def process(text):
    list = text.split('>>')
    result = ''
    for str in list:
        result += str.rstrip()
    return result


dataset_name = 'Books'
n_item = 86
n_user = 47
# item 85 21
# user 46 84
item_path = f'data/portrait/{dataset_name}/portrait_text/item_portrait.json'
user_path = f'data/portrait/{dataset_name}/portrait_text/user_portrait.json'

item_save_path = f'data/portrait/{dataset_name}/portrait_embedding/item_portrait_embedding.json'
user_save_path = f'data/portrait/{dataset_name}/portrait_embedding/user_portrait_embedding.json'

if not os.path.exists(item_path + '.json'):
    result = {}
    for i in range(n_item):
        path = item_path + f'_{i}.json'
        with open(path, 'r') as f:
            items = json.load(f)
        for item in items.items():
            key = item[0]
            portrait = item[1]
            result[key] = portrait
    with open(item_path + '.json', 'w') as f:
        json.dump(result, f, indent=4)
if not os.path.exists(user_path + '.json'):
    result = {}
    for i in range(n_user):
        path = user_path + f'_{i}.json'
        with open(path, 'r') as f:
            users = json.load(f)
        for user in users.items():
            key = user[0]
            portrait = user[1]
            result[key] = portrait
    with open(user_path + '.json', 'w') as f:
        json.dump(result, f, indent=4)

with open(item_path + '.json', 'r') as f:
    data = json.load(f)
    vo = voyage()
    model = 'voyage-large-2-instruct'
    result = {}
    for item in data.items():
        iid = item[0]
        text = item[1]
        text_processed = text#process(text)
        print(iid)
        embedding = vo.get_embedding(text=text_processed, model=model)
        result[iid] = embedding[0]
    with open(item_save_path, 'w') as f:
        json.dump(result, f, indent=4)
with open(user_path + '.json', 'r') as f:
    data = json.load(f)
    vo = voyage()
    model = 'voyage-large-2-instruct'
    result = {}
    for item in data.items():
        uid = item[0]
        text = item[1]
        text_processed = text#process(text)
        print(uid)
        embedding = vo.get_embedding(text=text_processed, model=model)
        result[uid] = embedding[0]
    with open(user_save_path, 'w') as f:
        json.dump(result, f, indent=4)
