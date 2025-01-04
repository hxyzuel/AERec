import json
import pickle

dataset_name = 'Books'  # Video_Games TripAdvisor movie_and_tv Books
path1 = f'data/data_processed_lilei/{dataset_name}/reviews.pickle'
save_path = f'data/data_for_lilei/{dataset_name}/reviews.pickle'
path_train = f'data/data_for_lilei/{dataset_name}/1/train.index'
path_test = f'data/data_for_lilei/{dataset_name}/1/test.index'
path_valid = f'data/data_for_lilei/{dataset_name}/1/validation.index'

with open(path1, 'rb') as f:
    data = pickle.load(f)

result = []
for item in data:
    temp = {}

    temp['user'] = item['user']
    temp['item'] = item['item']
    temp['rating'] = item['rating']
    temp['predicted'] = 'book'

    if 'sentence' in item:
        temp['template'] = item['sentence'][0]
    else:
        temp['template'] = ('book', 'good', 'great book!' + item['text'], 1)
    result.append(temp)

with open(save_path, 'wb') as file:
    pickle.dump(result, file)

total_length = len(result)
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
test_list = index_list

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
