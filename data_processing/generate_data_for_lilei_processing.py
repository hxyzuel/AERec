import json
import pickle

dataset_name = 'movie_and_tv'  # Video_Games TripAdvisor movie_and_tv
for dataset_name in ['Books']:
    path = f'data/data_for_lilei/{dataset_name}/reviews.pickle'
    data_path = f'data/data_after_sparsity_processing/{dataset_name}/dataAfter_{dataset_name}.json'
    save_path = f'data/data_for_lilei2/{dataset_name}/{dataset_name}.json'

    with open(path, 'rb') as f:
        data = pickle.load(f)
    with open(data_path, 'r') as f:
        data1 = json.load(f)

    dict = {}
    for item in data1:
        u_id = item['reviewerID']
        i_id = item['asin']
        dict[u_id + '_' + i_id] = item

    result = []
    for item in data:
        u_id = item['user']
        i_id = item['item']
        key = u_id + '_' + i_id

        item1 = dict[key]
        reviewText = item1['reviewText'].split('.')[0].strip()
        if 'summary' in item1:
            summary = item1['summary']
        else:
            summary = 'good'

        u_name = 'user'
        helpful = [0, 0]
        # reviewText = item['template'][2]
        overall = 5.0
        # summary = item['template'][0]
        unixReviewTime = 1393545600
        reviewTime = '02 28, 2014'
        temp = {}
        temp['reviewerID'] = u_id
        temp['asin'] = i_id
        temp['reviewerName'] = u_name
        temp['overall'] = overall
        temp['summary'] = summary
        temp['unixReviewTime'] = unixReviewTime
        temp['reviewTime'] = reviewTime
        temp['reviewText'] = reviewText
        result.append(temp)

    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4)
