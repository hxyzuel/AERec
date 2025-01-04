import json
import pickle

dataset_name = 'Books'  # TripAdvisor Video_Games movie_and_tv Books
rewrite_path = f'data/rewrite_review/{dataset_name}/rewrite_review.json'
save_path = f'data/data_for_lilei/{dataset_name}/reviews.pickle'
path_train = f'data/data_for_lilei/{dataset_name}/1/train.index'
path_test = f'data/data_for_lilei/{dataset_name}/1/test.index'
path_valid = f'data/data_for_lilei/{dataset_name}/1/validation.index'

result_list = []
n = 0
with open(rewrite_path, 'r') as f:
    data = json.load(f)
length = len(data)
for item in data:
    temp = {}
    temp['user'] = item['user']
    temp['item'] = item['item']
    temp['rating'] = 5
    temp['template'] = (item['aspects'][0], 'good', item['descrips'][0], 1)
    temp['predicted'] = item['aspects'][0]
    result_list.append(temp)
    # if n % 3 == 0:
    #     neg_idx = random.randint(0, length - 1)
    #     neg = {}
    #     neg['user'] = item['user']
    #     neg['item'] = data[neg_idx]['item']
    #     neg['rating'] = 1
    #     neg['template'] = (data[neg_idx]['gt_aspects'][0], 'bad', data[neg_idx]['gt_descrips'][0], -1)
    #     neg['predicted'] = data[neg_idx]['gt_aspects'][0]
    #     result_list.append(neg)
    n += 1

with open(save_path, 'wb') as file:
    pickle.dump(result_list, file)

total_length = len(result_list)
print(total_length)
index_list = []
for i in range(total_length):
    index_list.append(i)
train_size = int(total_length * 0.8)
val_size = int(total_length * 0.1)
test_size = total_length - train_size - val_size

train_list = index_list[:train_size]
val_list = index_list[train_size:train_size + val_size]
# test_list = index_list[train_size + val_size:]
test_list=index_list

with open(path_train, 'w') as f:
    f.write(str(train_list[0]))
    for idx in train_list[1:]:
        f.write(' ' + str(idx))
with open(path_test, 'w') as f:
    f.write(str(test_list[0]))
    for idx in test_list[1:]:
        f.write(' ' + str(idx))
with open(path_valid, 'w') as f:
    f.write(str(val_list[0]))
    for idx in val_list[1:]:
        f.write(' ' + str(idx))
