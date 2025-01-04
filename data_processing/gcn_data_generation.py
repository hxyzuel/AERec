import json
import random

dataset_name = 'Books'
rating_index = 'rating' if dataset_name == 'image_review_all' else 'overall'
id_index = 'user_id' if dataset_name == 'image_review_all' else 'reviewerID'
asin_index = 'business_id' if dataset_name == 'image_review_all' else 'asin'
txt_index = 'review_text' if dataset_name == 'image_review_all' else 'reviewText'

with open(f"data/data_sort_by_time/{dataset_name}/user_prompt_{dataset_name}.json", 'r') as json_f:
    data = json.load(json_f)

user_mapped_id = 0
item_mapped_id = 0

user_id_map = {}
item_id_map = {}

mapped_hist = {}

for item in data.items():
    user = item[0]
    hist_list = item[1]
    hist_mapped = []
    for hist in hist_list:
        item = hist[asin_index]
        if item not in item_id_map:
            item_id_map[str(item)] = str(item_mapped_id)
            hist_mapped.append(str(item_mapped_id))
            item_mapped_id += 1
        else:
            hist_mapped.append(item_id_map[str(item)])

    if user not in user_id_map:
        user_id_map[str(user)] = str(user_mapped_id)
        hist_mapped = list(set(hist_mapped))
        mapped_hist[str(user_mapped_id)] = hist_mapped
        user_mapped_id += 1
    else:
        hist_mapped = list(set(hist_mapped))
        mapped_hist[str(user_id_map[str(user)])] = hist_mapped
# for item in data.items():
#     user = item[0]
#     hist = item[1]
#     hist_mapped = []
#     for item in hist:
#         if item not in item_id_map:
#             item_id_map[str(item)] = str(item_mapped_id)
#             hist_mapped.append(item_mapped_id)
#             item_mapped_id += 1
#         else:
#             hist_mapped.append(item_id_map[str(item)])
#     if user not in user_id_map:
#         user_id_map[str(user)] = user_mapped_id
#         mapped_hist[str(user_mapped_id)] = hist_mapped
#         user_mapped_id += 1
#     else:
#         mapped_hist[str(user_id_map[str(user)])] = hist_mapped

with open(f'data/data_gcn/{dataset_name}/user_list2.txt', 'w') as file:
    file.write('org_id remap_id\n')
    for user in user_id_map.items():
        file.write(f'{user[0]} {user[1]}\n')

with open(f'data/data_gcn/{dataset_name}/item_list2.txt', 'w') as file:
    file.write('org_id remap_id\n')
    for item in item_id_map.items():
        file.write(f'{item[0]} {item[1]}\n')

test_ratio = 0.2

with open(f'data/data_gcn/{dataset_name}/total_interaction.txt', 'w') as file_total:
    with open(f'data/data_gcn/{dataset_name}/train.txt', 'w') as file_train:
        with open(f'data/data_gcn/{dataset_name}/test.txt', 'w') as file_test:
            for item in mapped_hist.items():
                user = item[0]
                hist = item[1]
                hist = list(set(hist))
                num_to_test = int(len(hist) * test_ratio)

                test_idxs = [random.randint(0, len(hist) - 1) for _ in range(num_to_test)]
                test = []
                train = []
                for i in range(len(hist)):
                    if i in test_idxs:
                        test.append(hist[i])
                    else:
                        train.append(hist[i])

                test = list(set(test))
                train = list(set(train))
                if user=='2':
                    print(hist)

                total_to_txt = str(user)
                test_to_txt = str(user)
                train_to_txt = str(user)
                for item in hist:
                    total_to_txt += f' {item}'
                total_to_txt += '\n'
                for item in test:
                    test_to_txt += f' {item}'
                test_to_txt += '\n'
                for item in train:
                    train_to_txt += f' {item}'
                train_to_txt += '\n'
                file_total.write(total_to_txt)
                file_test.write(test_to_txt)
                file_train.write(train_to_txt)

with open(f'data/data_gcn/{dataset_name}/train.txt', 'r') as f:
    lines = f.readlines()
to_txt = ''
for line in lines:
    temp = line.rstrip().split(' ')
    user = temp[0]
    hist = temp[1:]
    hist = list(set(hist))
    to_txt += user
    for item in hist:
        to_txt += f' {item}'
    to_txt += '\n'
with open(f'data/data_gcn/{dataset_name}/train1.txt', 'w') as f1:
    f1.write(to_txt)

with open(f'data/data_gcn/{dataset_name}/test.txt', 'r') as f2:
    lines = f2.readlines()
to_txt = ''
for line in lines:
    temp = line.rstrip().split(' ')
    user = temp[0]
    hist = temp[1:]
    hist = list(set(hist))
    to_txt += user
    for item in hist:
        to_txt += f' {item}'
    to_txt += '\n'
with open(f'data/data_gcn/{dataset_name}/test1.txt', 'w') as f3:
    f3.write(to_txt)
