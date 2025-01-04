import json
import re
import pandas as pd
import numpy as np
import os
import pickle


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def numerize(tp):
    uid = map(lambda x: user2id[x], tp['user_id'])
    sid = map(lambda x: item2id[x], tp['item_id'])
    tp['user_id'] = list(uid)
    tp['item_id'] = list(sid)
    return tp


def numerize_meta(tp, item2id):
    sid = []
    for idx, term in enumerate(tp['item_id']):
        if term in item2id.keys():
            sid.append(item2id[term])
        else:
            sid.append(-1)
    tp['item_id'] = sid
    return tp


dataset_name = 'Books'  # movie_and_tv Video_Games TripAdvisor
data_path = f'data/data_after_sparsity_processing/{dataset_name}/dataAfter_{dataset_name}.json'
meta_path2 = f'data/data_after_sparsity_processing/{dataset_name}/meta_{dataset_name}.json'

TPS_DIR = f'data/data_for_emer/{dataset_name}/split_data/'
META_TPS_DIR = f'data/data_for_emer/{dataset_name}/'

with open(data_path, 'r') as f:
    data1 = json.load(f)

dict_og = {}
if dataset_name == 'TripAdvisor':
    dict1 = {}
    dict2 = {}
    path = 'data/data_after_sparsity_processing/TripAdvisor/OriginalReviews.json'
    with open(path, 'r') as f:
        data = json.load(f)
        for item in data:
            dict1[item['hotelID']] = item['hotelTitle']
            dict2[item['hotelID']] = 'Hotel'
            uid = item['userID']
            iid = item['hotelID']
            text = ''
            if 'reviewHeading' not in item:
                text = clean_str(item['reviewText'].split('.')[0].strip())
            else:
                text = clean_str(item['reviewHeading'].strip())
            dict_og[uid + '_' + iid] = text
else:
    dict1 = {}
    with open(meta_path2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            data = json.loads(line)
            dict1[data['asin']] = data['title']
    dict2 = {}
    with open(meta_path2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            data = json.loads(line)
            dict2[data['asin']] = data['category']

review_uid = []
review_iid = []
review_rating = []
review_tip = []
review_review = []

meta_iid = []
meta_cate = []
meta_des = []
meta_title = []

for item in data1:
    if dataset_name == 'TripAdvisor':
        iid = item['asin']
        review_uid.append(item['reviewerID'])
        review_iid.append(iid)
        review_rating.append(item['overall'])
        review_review.append(item['reviewText'])
        review_tip.append(dict_og[item['reviewerID']+'_'+iid])
    else:
        iid = item['asin']
        review_uid.append(item['reviewerID'])
        review_iid.append(iid)
        review_rating.append(item['overall'])
        review_review.append(item['reviewText'])
        if 'summary' not in item:
            review_tip.append(clean_str(item['reviewText'].split('.')[0].strip()))
        else:
            review_tip.append(clean_str(item['summary'].strip()))
        # review_tip.append(clean_str(item['reviewText'].split('.')[0].strip()))

    if iid not in meta_iid:
        meta_iid.append(iid)
        meta_des.append('des')
        try:
            meta_cate.append(dict2[iid])
        except:
            if dataset_name == 'Video_Games':
                meta_cate.append(['Video Games'])
            elif dataset_name == 'TripAdvisor':
                meta_cate.append(['Hotel'])
            else:
                meta_cate.append(['a book'])
        try:
            title = dict1[iid]
        except:
            if dataset_name == 'Video_Games':
                title = 'A Video Game'
            elif dataset_name == 'TripAdvisor':
                title = 'A Hotel'
            else:
                title = 'A Book'
        meta_title.append(title)

meta_data = pd.DataFrame({'item_id': pd.Series(meta_iid),
                          'categories': pd.Series(meta_cate),
                          'description': pd.Series(meta_des),
                          'title': pd.Series(meta_title)})[['item_id', 'categories', 'description', 'title']]

data = pd.DataFrame({'user_id': pd.Series(review_uid),
                     'item_id': pd.Series(review_iid),
                     'ratings': pd.Series(review_rating),
                     'summary': pd.Series(review_tip),
                     'reviews': pd.Series(review_review), })[
    ['user_id', 'item_id', 'ratings', 'summary', 'reviews']]

meta_data.to_csv(os.path.join(TPS_DIR, dataset_name + '_infor.csv'), index=False)
meta_data.to_csv(os.path.join(TPS_DIR, dataset_name + '_infor.csv'), index=False,
                 header=['item_id', 'categories', 'description', 'title'])
usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')
uid_list = usercount['user_id'].tolist()
# unique_uid, unique_sid = usercount.index, itemcount.index
# for x in enumerate(unique_uid):
#     print(x)

uid_index_list = usercount.index.tolist()
uid_list = usercount['user_id'].tolist()
iid_index_list = itemcount.index.tolist()
iid_list = itemcount['item_id'].tolist()

# user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
# item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
user2id = dict(zip(uid_list, uid_index_list))
item2id = dict(zip(iid_list, iid_index_list))

data = numerize(data)
meta_data = numerize_meta(meta_data, item2id)
meta_data = meta_data[~meta_data['item_id'].isin([-1])]

n_ratings = data.shape[0]
test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True
tp_test_valid = data[test_idx]
tp_train = data[~test_idx]

n_ratings = tp_test_valid.shape[0]
test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True
tp_valid = tp_test_valid[~test_idx]
tp_test = tp_test_valid[test_idx]

tp_train.to_csv(os.path.join(TPS_DIR, dataset_name + '_train.csv'), index=False, header=None)
tp_valid.to_csv(os.path.join(TPS_DIR, dataset_name + '_valid.csv'), index=False, header=None)
tp_test.to_csv(os.path.join(TPS_DIR, dataset_name + '_test.csv'), index=False, header=None)
meta_data.to_csv(os.path.join(TPS_DIR, dataset_name + '_infor_meta.csv'), index=False, header=None)
pickle.dump(user2id, open(os.path.join(TPS_DIR, 'user2id'), 'wb'))
pickle.dump(item2id, open(os.path.join(TPS_DIR, 'item2id'), 'wb'))
