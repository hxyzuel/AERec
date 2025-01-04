import json

dataset_name = 'Books'

id_index = 'user_id' if dataset_name == 'image_review_all' else 'reviewerID'
asin_index = 'business_id' if dataset_name == 'image_review_all' else 'asin'
txt_index = 'review_text' if dataset_name == 'image_review_all' else 'reviewText'

item_list_path = f'data/data_for_review_rewrite/{dataset_name}/item_list.txt'
user_list_path = f'data/data_for_review_rewrite/{dataset_name}/user_list.txt'
dict_path = f'data/data_for_review_rewrite/{dataset_name}/dict_{dataset_name}.json'
test_set_path = f'data/data_for_review_rewrite/{dataset_name}/test.txt'

data_for_rewrite_path = f'data/data_for_review_rewrite/{dataset_name}/data_for_rewrite_review.json'

item_dict = {}
with open(item_list_path, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        temp = line.rstrip().split(' ')
        raw_id = temp[0]
        id = temp[1]
        item_dict[id] = raw_id
user_dict = {}
with open(user_list_path, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        temp = line.rstrip().split(' ')
        raw_id = temp[0]
        id = temp[1]
        user_dict[id] = raw_id

with open(dict_path, 'r') as f:
    dict = json.load(f)

result = {}
with open(test_set_path, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        temp = line.rstrip().split(' ')
        uid = temp[0]
        raw_uid = user_dict[uid]
        for iid in temp[1:]:
            raw_iid = item_dict[iid]
            key = raw_uid + '_' + raw_iid
            review = dict[key][txt_index]
            result[key] = review

with open(data_for_rewrite_path, 'w') as f:
    json.dump(result, f, indent=4)
