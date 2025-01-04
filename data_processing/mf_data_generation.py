import json
import pandas as pd

dataset_name = 'Books'
# rating_index = 'rating' if dataset_name == 'image_review_all' else 'overall'
rating_index = 'rating'
id_index = 'user_id' if dataset_name == 'image_review_all' else 'reviewerID'
asin_index = 'business_id' if dataset_name == 'image_review_all' else 'asin'
txt_index = 'review_text' if dataset_name == 'image_review_all' else 'reviewText'

train_path = f'data/data_gcn/{dataset_name}/train.txt'
user_list_path = f'data/data_gcn/{dataset_name}/user_list.txt'
item_list_path = f'data/data_gcn/{dataset_name}/item_list.txt'
dict_save_path = f'data/data_mf/{dataset_name}/dict_{dataset_name}.json'
ratings_path = f'data/data_mf/{dataset_name}/ratings.dat'
data_after_path = f'data/data_after_sparsity_processing/{dataset_name}/dataAfter_{dataset_name}.json'
# def to_dict(data_after_path, dict_save_path):
#     with open(data_after_path, 'r') as f:
#         file = json.load(f)
#
#     dict = {}
#
#     for item in file:
#         value = {}
#         value['rating'] = item[rating_index]
#         value['reviewText'] = item[txt_index]
#         key = item[id_index] + '_' + item[asin_index]
#         dict[key] = value
#     with open(dict_save_path, 'w') as json_file:
#         json.dump(dict, json_file)
#
# to_dict(data_after_path,dict_save_path)

user_dict = {}
with open(user_list_path, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        temp = line.rstrip().split(' ')
        raw_id = temp[0]
        id = temp[1]
        user_dict[id] = raw_id

item_dict = {}
with open(item_list_path, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        temp = line.rstrip().split(' ')
        raw_id = temp[0]
        id = temp[1]
        item_dict[id] = raw_id

with open(dict_save_path, 'r') as f:
    dict = json.load(f)

result = {}
user_list = []
item_list = []
rating_list = []
with open(train_path, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        temp = line.rstrip().split(' ')
        uid = temp[0]
        u_raw_id = user_dict[uid]
        for iid in temp[1:]:
            i_raw_id = item_dict[iid]
            rating = dict[u_raw_id + '_' + i_raw_id][rating_index]
            user_list.append(uid)
            item_list.append(iid)
            rating_list.append(float(rating))

result['user'] = user_list
result['item'] = item_list
result['rating'] = rating_list
df = pd.DataFrame(result)
df.to_csv(ratings_path, sep='\t', index=False)
