import re
import numpy as np
import json
from collections import Counter
import matplotlib.pyplot as plt


def count_english_words(text):
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return len(words)


def compute(dataset_name, min, max):
    file_path = f'data/data_raw/{dataset_name}/{dataset_name}.json'
    fin = open(file_path, 'r')
    listItem = []
    listUser = []
    averageReview = []
    val = ""
    helpLost = 0.0
    i = 0
    for line in fin:
        # df = eval(line)
        print(line)
        df = json.loads(line)
        if txt_index not in df:
            continue
        averageReview.append(count_english_words(df[txt_index]))
        if (count_english_words(df[txt_index]) >= min) & (count_english_words(df[txt_index]) <= max):
            listUser.append(df[id_index])
            listItem.append(df[asin_index])
    countUser = dict(Counter(listUser))
    countItem = dict(Counter(listItem))
    json.dump(countUser, open(f"data/data_after_sparsity_processing/{dataset_name}/countUser_{dataset_name}.json", "w"))
    json.dump(countItem, open(f"data/data_after_sparsity_processing/{dataset_name}/countItem_{dataset_name}.json", "w"))
    json.dump(averageReview,
              open(f"data/data_after_sparsity_processing/{dataset_name}/averageReview_{dataset_name}.json", "w"))


def drawReviewLen(dataset_name):
    file_path1 = f"data/data_after_sparsity_processing/{dataset_name}/averageReview_{dataset_name}.json"
    with open(file_path1, 'r') as f:
        reviewData = json.load(f)

    # Sort the selected items into 10 bins based on their value
    data = np.random.choice(reviewData, 10000, replace=False)
    sample = data[data < 500]
    # 计算抽样中不同值的分布情况
    values, counts = np.unique(sample, return_counts=True)

    # Create the bar chart
    plt.rcParams['font.sans-serif'] = ['kaiti']
    plt.figure(figsize=[20, 6])
    plt.title('1W条review长度汇总')
    plt.xlabel('长度范围')
    plt.ylabel('数量')
    plt.plot(values, counts, marker='.', linestyle='-', color='b')
    plt.grid(axis='y')
    plt.show()


def getUser(dataset_name, min):
    file_path1 = f"data/data_after_sparsity_processing/{dataset_name}/countUser_{dataset_name}.json"
    with open(file_path1, 'r') as f:
        userData = json.load(f)
    goodUser = {key: val for key, val in userData.items() if val > min}
    # print(len(goodUser)) #3919
    return goodUser


def getItem(dataset_name, min):
    file_path1 = f"data/data_after_sparsity_processing/{dataset_name}/countItem_{dataset_name}.json"
    with open(file_path1, 'r') as f:
        userData = json.load(f)
    goodItem = {key: val for key, val in userData.items() if val > min}
    # print(len(goodUser)) #3919
    return goodItem


def drawUserCount(dataset_name, min, draw_max):
    user = getUser(dataset_name, min)  # 返回合格user字典
    value = []
    for key, val in user.items():
        value.append(val)

    sample = [temp for temp in value if temp < draw_max]
    # Sort the selected items into 10 bins based on their value
    # 计算抽样中不同值的分布情况
    values, counts = np.unique(sample, return_counts=True)

    # Create the bar chart
    plt.rcParams['font.sans-serif'] = ['kaiti']
    plt.figure(figsize=[20, 6])
    plt.plot(values, counts, marker='.', linestyle='-', color='b')
    plt.title('用户评论数量汇总（10+）')
    plt.xlabel('评论条数')
    plt.ylabel('数量')
    plt.grid(axis='y')
    plt.show()


def drawItemCount(dataset_name, min, draw_max):
    user = getItem(dataset_name, min)  # 返回合格user字典
    value = []
    for key, val in user.items():
        value.append(val)

    sample = [temp for temp in value if temp < draw_max]
    # Sort the selected items into 10 bins based on their value
    # 计算抽样中不同值的分布情况
    values, counts = np.unique(sample, return_counts=True)

    # Create the bar chart
    plt.rcParams['font.sans-serif'] = ['kaiti']
    plt.figure(figsize=[20, 6])
    plt.plot(values, counts, marker='.', linestyle='-', color='r')
    plt.title('项目被评论数量汇总（5+）')
    plt.xlabel('评论条数')
    plt.ylabel('数量')
    plt.grid(axis='y')
    plt.show()


def draw(dataset_name, max):
    file_path1 = f"data/data_after_sparsity_processing/{dataset_name}/dataAfter_{dataset_name}.json"
    listUser = []
    listItem = []
    countUser = {}
    countItem = {}
    with open(file_path1, 'r') as f:
        data = json.load(f)
    for line in data:
        listUser.append(line[id_index])
        listItem.append(line[asin_index])
    countUser = dict(Counter(listUser))
    countItem = dict(Counter(listItem))
    User = [temp for temp in countUser.values() if (temp < max)]
    Item = [temp for temp in countItem.values() if (temp < max)]
    itemvalues, itemcount = np.unique(Item, return_counts=True)

    # Create the bar chart
    plt.rcParams['font.sans-serif'] = ['kaiti']
    plt.figure(figsize=[20, 6])
    plt.plot(itemvalues, itemcount, marker='.', linestyle='-', color='r')
    plt.title('用户b/项目r评论数量汇总（5+）')
    plt.xlabel('评论条数')
    plt.ylabel('数量')
    plt.grid(axis='y')
    values, counts = np.unique(User, return_counts=True)
    # Create the bar chart
    plt.plot(values, counts, marker='.', linestyle='-', color='b')
    plt.show()


def washData(dataset_name, min_user, min_item, min_wash, max_wash):
    file_path = f"data/data_raw/{dataset_name}/{dataset_name}.json"
    fin = open(file_path, 'r')

    user = getUser(dataset_name, min_user)
    userId = []
    data = []
    for key, val in user.items():
        userId.append(key)
    item = getItem(dataset_name, min_item)
    itemId = []
    for key, val in item.items():
        itemId.append(key)

    for line in fin:
        df = json.loads(line)
        if txt_index not in df:
            continue
        leng = count_english_words(df[txt_index])
        if (leng < min_wash) | (leng > max_wash):
            continue
        elif (df[id_index] not in userId) | (df[asin_index] not in itemId):
            continue
        else:
            data.append(df)
    # print(len(data))  #229702
    json.dump(data, open(f"data/data_after_sparsity_processing/{dataset_name}/dataAfter10_{dataset_name}.json", "w"))


def wash(dataset_name, min_user, min_item):
    listUser = []
    listItem = []
    data = []
    result = []
    file_path1 = f"data/data_after_sparsity_processing/{dataset_name}/dataAfter10_{dataset_name}.json"
    with open(file_path1, 'r') as f:
        fin = json.load(f)
    i = 0
    for line in fin:
        listUser.append(line[asin_index])
    countUser = dict(Counter(listUser))
    good = [key for key, val in countUser.items() if (val > min_user)]
    for line in fin:
        if line[asin_index] not in good:
            continue
        else:
            data.append(line)
    for line in data:
        listItem.append(line[id_index])
    countItem = dict(Counter(listItem))
    Item = [key for key, val in countItem.items() if (val > min_item)]
    for line in data:
        if line[id_index] not in Item:
            continue
        else:
            result.append(line)
    print(len(result))  # 128687
    json.dump(result, open(f"data/data_after_sparsity_processing/{dataset_name}/dataAfter_{dataset_name}.json", "w"))


def findhis(dataset_name):
    file_path1 = f"data/data_after_sparsity_processing/{dataset_name}/dataAfter_{dataset_name}.json"
    with open(file_path1, 'r') as f:
        userData = json.load(f)
    result = {}
    for purchase in userData:
        user = purchase[id_index]
        item = purchase[asin_index]
        if user in result:
            if item not in result[user]:
                result[user].append(item)
        else:
            result[user] = [item]

    json.dump(result, open(f"data/data_after_sparsity_processing/{dataset_name}/userHistory_{dataset_name}.json", "w"))


dataset_name = 'Books'  # 12 40 180

rating_index = 'rating' if dataset_name == 'image_review_all' else 'overall'
id_index = 'user_id' if dataset_name == 'image_review_all' else 'reviewerID'
asin_index = 'business_id' if dataset_name == 'image_review_all' else 'asin'
txt_index = 'review_text' if dataset_name == 'image_review_all' else 'reviewText'

min_in_compute = 60
max_in_compute = 180

min_in_getUser = 30
min_in_getItem = 30

max_in_draw_xxx = 200

min_in_washData = 60
max_in_washData = 180

filtered_path = f'data/data_raw/{dataset_name}/{dataset_name}1.json'
raw_path = f'data/data_raw/{dataset_name}/{dataset_name}.json'
# with open(filtered_path, 'r') as fl:
#     data = json.load(fl)
# with open(raw_path, 'w') as fr:
#     for item in data:
#         temp = {}
#         temp['overall'] = item['overall']
#         temp['reviewTime'] = item['reviewTime']
#         temp['reviewerID'] = item['reviewerID']
#         temp['asin'] = item['asin']
#         temp['reviewText'] = item['reviewText']
#         if 'summary' not in item:
#             continue
#         temp['summary'] = item['summary']
#         temp['unixReviewTime'] = item['unixReviewTime']
#         fr.write(json.dumps(temp) + '\n')

# compute(dataset_name, min_in_compute, max_in_compute)
# washData(dataset_name, min_in_getUser, min_in_getItem, min_in_washData, max_in_washData)
# wash(dataset_name, min_in_getUser, min_in_getItem)
# findhis(dataset_name)

file_path1 = f"data/data_after_sparsity_processing/{dataset_name}/dataAfter_{dataset_name}.json"
listItem = []
listUser = []
with open(file_path1, 'r') as f:
    fin = json.load(f)
countRe = len(fin)
print(countRe)
for i in range(len(fin)):
    df = fin[i]
    listItem.append(df["asin"])
countItem = dict(Counter(listItem))
print(len(countItem))
for i in range(len(fin)):
    df = fin[i]
    listUser.append(df["reviewerID"])
countUser = dict(Counter(listUser))
print(len(countUser))

len_sum = 0
n = 0
for item in fin:
    text = item['reviewText']
    n += 1
    len_sum += count_english_words(text)

print(len_sum / n)


def sparity(interaction, user, item):
    return 1 - interaction / user / item


#
print(sparity(countRe, len(countUser), len(countItem)))
