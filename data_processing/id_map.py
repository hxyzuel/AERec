import json
import random

dataset='Books'
with open(f'data/data_after_sparsity_processing/{dataset}/userHistory_{dataset}.json', 'r') as json_f:
    data = json.load(json_f)

user_mapped_id = 0
item_mapped_id = 0

user_id_map = {}
item_id_map = {}

mapped_hist = {}

for item in data.items():
    user = item[0]
    hist = item[1]
    hist_mapped = []
    for item in hist:
        if item not in item_id_map:
            item_id_map[str(item)] = str(item_mapped_id)
            hist_mapped.append(item_mapped_id)
            item_mapped_id += 1
        else:
            hist_mapped.append(item_id_map[str(item)])
    if user not in user_id_map:
        user_id_map[str(user)] = user_mapped_id
        mapped_hist[str(user_mapped_id)] = hist_mapped
        user_mapped_id += 1
    else:
        mapped_hist[str(user_id_map[str(user)])] = hist_mapped

with open(f'data/data_gcn/{dataset}/user_list.txt', 'w') as file:
    file.write('org_id remap_id\n')
    for user in user_id_map.items():
        file.write(f'{user[0]} {user[1]}\n')

with open(f'data/data_gcn/{dataset}/item_list.txt', 'w') as file:
    file.write('org_id remap_id\n')
    for item in item_id_map.items():
        file.write(f'{item[0]} {item[1]}\n')

test_ratio = 0.2

with open(f'data/data_gcn/{dataset}/total_interaction.txt', 'w') as file_total:
    with open(f'data/data_gcn/{dataset}/train.txt', 'w') as file_train:
        with open(f'data/data_gcn/{dataset}/test.txt', 'w') as file_test:
            for item in mapped_hist.items():
                user = item[0]
                hist = item[1]
                num_to_test = int(len(hist) * test_ratio)
                test = random.sample(hist, num_to_test)
                train = [x for x in hist if x not in test]
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

# 将画像的文本嵌入存为json字典格式
# 原始ID : 嵌入向量
