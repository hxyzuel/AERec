import json

dataset_name = 'Books'

data_path = f'data/data_after_sparsity_processing/{dataset_name}/dataAfter_{dataset_name}.json'
dict_save_path = f'data/data_mf/{dataset_name}/dict_{dataset_name}.json'

with open(data_path, 'r') as f:
    data = json.load(f)

dict = {}
for item in data:
    uid = item['reviewerID']
    iid = item['asin']
    key = uid + '_' + iid

    value = {}
    rating = item['overall']
    text = item['reviewText']
    value['rating'] = rating
    value['reviewText'] = text

    dict[key] = value

with open(dict_save_path, 'w') as f:
    json.dump(dict, f, indent=4)
