import json
import csv
import re
import pickle


def skip_invalid(aspects, descrips):
    # 去除无效解释文本，如包含not enough
    new_aspects = []
    new_descrips = []
    for a, d in zip(aspects, descrips):
        if (d == '' or 'Not applicable' in d or 'not applicable' in d or 'No specific' in d
                or 'no specific' in d or 'Not mentioned' in d or 'not mentioned' in d or 'No mention' in d
                or 'no mention' in d or 'Not mention' in d or 'not mention' in d or 'no inferences' in d
                or 'No inferences' in d or 'Not enough' in d or 'not enough' in d or 'No information' in d):
            continue

        pattern = r'no\s+(\w*)\s+mention'
        match = re.search(pattern, d)
        if match:
            content_between = match.group(1)
            if content_between:
                continue

        pattern = r'No\s+(\w*)\s+mention'
        match = re.search(pattern, d)
        if match:
            content_between = match.group(1)
            if content_between:
                continue

        pattern = r'not\s+(\w*)\s+mentioned'
        match = re.search(pattern, d)
        if match:
            content_between = match.group(1)
            if content_between:
                continue

        pattern = r'Not\s+(\w*)\s+mentioned'
        match = re.search(pattern, d)
        if match:
            content_between = match.group(1)
            if content_between:
                continue

        pattern = r'not\s+(\w*)\s+mention'
        match = re.search(pattern, d)
        if match:
            content_between = match.group(1)
            if content_between:
                continue

        pattern = r'Not\s+(\w*)\s+mention'
        match = re.search(pattern, d)
        if match:
            content_between = match.group(1)
            if content_between:
                continue

        pattern = r'not\s+(\w*)\s+enough'
        match = re.search(pattern, d)
        if match:
            content_between = match.group(1)
            if content_between:
                continue

        pattern = r'Not\s+(\w*)\s+enough'
        match = re.search(pattern, d)
        if match:
            content_between = match.group(1)
            if content_between:
                continue

        new_aspects.append(a)
        new_descrips.append(d)
    return new_aspects, new_descrips


def if_contain(aspects, descrips):
    # 保留aspect具体词出现在对应描述文本中的条目
    new_aspects = []
    new_descrips = []
    if len(aspects) == 1:
        return aspects, descrips
    else:
        for a, d in zip(aspects, descrips):
            if a.lower() in d.lower():
                new_aspects.append(a)
                new_descrips.append(d)
        if len(new_aspects) == 0:
            new_aspects.append(aspects[0])
            new_descrips.append(descrips[0])
    return new_aspects, new_descrips


def sort(dict, aspects, descrips):
    # 将aspect和对应描述按aspect出现的频率排列
    sorted_list = sorted(dict.items(), key=lambda x: x[1])
    sorted_list.reverse()
    new_list = []
    for item in sorted_list:
        new_list.append(item[0])
    new_aspects = []
    new_descrips = []
    for aa in new_list:
        if aa in aspects:
            index = aspects.index(aa)
            new_aspects.append(aspects[index])
            new_descrips.append(descrips[index])
    return new_aspects, new_descrips


def get_aspects(text):
    aspects = []
    descrips = []
    if text[:2] != '>>':
        text = '>>' + text
    list = text.split('\n')
    # list = text.split('>>')
    for i in list:
        if i == '':
            continue
        temp = i.split(":")
        if len(temp) == 1:
            continue
        aspect = temp[0]
        descrip = ''
        for j in temp[1]:
            descrip += j
        descrip = descrip.rstrip().lstrip()
        aspects.append(aspect)
        descrips.append(descrip)
    return aspects, descrips


dataset_name = "Books"  # movie_and_tv TripAdvisor Video_Games
rewrite_path = f'data/rewrite_review/{dataset_name}/rewrite_review.json'
rec_item_path = f'data/data_sort_by_time/{dataset_name}/rec_item_{dataset_name}.json'
aspect_path = f'data/aspects/{dataset_name}/aspects_full.txt'

save_path = f'data/data_for_lilei/{dataset_name}/reviews.pickle'
train_path = f'data/data_for_lilei/{dataset_name}/1/train.index'
test_path = f'data/data_for_lilei/{dataset_name}/1/test.index'
valid_path = f'data/data_for_lilei/{dataset_name}/1/validation.index'

aspect_list = []
with open(aspect_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        aspect_list.append(line.rstrip())

example = {}
if dataset_name == 'Books':
    example = {
        "Plot": 'The plot of the book is decent, which caters to the user`s reference.'
    }
elif dataset_name == 'movie_and_tv':
    example = {
        "Plot": 'The plot of the movie is decent, which caters to the user`s reference.'
    }
else:
    example = {
        "Comfort": 'The rooms are big and bright, offering a comfortable stay for guests, which caters to the user`s '
                   'consideration.'
    }

with open(rewrite_path, 'r') as f:
    rewrites = json.load(f)
with open(rec_item_path, 'r') as f:
    rec_items = json.load(f)

if dataset_name == 'movie_and_tv':
    rec = {}
    flag = False
    for item in rec_items.items():
        uid = item[0]
        iid = item[1]
        if uid == 'A1L46PC7ND70T9':
            flag = True
        if flag:
            rec[uid] = iid
    rec_items = rec
elif dataset_name == 'TripAdvisor':
    rec = {}
    flag = False
    for item in rec_items.items():
        uid = item[0]
        iid = item[1]
        if uid == '6BE2AD4D9BF13D5211B8A0242FCF2C3B':
            flag = True
        if flag:
            rec[uid] = iid
    rec_items = rec
elif dataset_name == 'Books':
    rec = {}
    flag = False
    for item in rec_items.items():
        uid = item[0]
        iid = item[1]
        if uid == 'AX4NG9BDHGE1S':
            flag = True
        if flag:
            rec[uid] = iid
    rec_items = rec

aspect_dict = {}
result_list = []
print(len(list(rec_items.items())))
for item in rec_items.items():
    uid = item[0]
    iid = item[1]
    if uid not in rewrites:
        continue
    rewrite = rewrites[uid]
    aspects, descrips = get_aspects(rewrite)
    if 'Game Series' in aspects:
        index = aspects.index('Game Series')
        aspects[index] = 'Games Series'
    for i in range(len(aspects)):
        aspects[i] = re.sub(r'^[^a-zA-Z]*|[^a-zA-Z]*$', '', aspects[i])
    aspects, descrips = skip_invalid(aspects, descrips)
    if len(aspects) == 0:
        aspects.append(list(example.keys())[0])
        descrips.append(example[list(example.keys())[0]])
    temp = {}
    temp['user'] = uid
    temp['item'] = iid
    temp['aspects'] = aspects
    temp['descrips'] = descrips
    result_list.append(temp)

    for aspect in aspects:
        if (aspect not in aspect_dict) and (aspect in aspect_list):
            aspect_dict[aspect] = 1
        elif (aspect in aspect_dict) and (aspect in aspect_list):
            aspect_dict[aspect] += 1

new_list = []
for item in result_list:
    temp = {}
    temp['user'] = item['user']
    temp['item'] = item['item']
    aspects, descrips = sort(aspect_dict, item['aspects'], item['descrips'])
    if len(aspects) == 0:
        aspects.append(list(example.keys())[0])
        descrips.append(example[list(example.keys())[0]])
    temp['aspects'] = aspects
    temp['descrips'] = descrips
    new_list.append(temp)

final_list = []
for item in new_list:
    temp = {}
    temp['user'] = item['user']
    temp['item'] = item['item']
    temp['rating'] = 5
    temp['template'] = (item['aspects'][0], 'good', item['descrips'][0], 1)
    temp['predicted'] = item['aspects'][0]
    final_list.append(temp)

with open(save_path, 'wb') as f:
    pickle.dump(final_list, f)

total_length = len(final_list)
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

with open(train_path, 'w') as f:
    f.write(str(train_list[0]))
    for idx in train_list[1:]:
        f.write(' ' + str(idx))
with open(test_path, 'w') as f:
    f.write(str(test_list[0]))
    for idx in test_list[1:]:
        f.write(' ' + str(idx))
with open(valid_path, 'w') as f:
    f.write(str(val_list[0]))
    for idx in val_list[1:]:
        f.write(' ' + str(idx))
