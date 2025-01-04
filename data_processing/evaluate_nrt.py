import json
import nltk

nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from utils import rouge_score, bleu_score, unique_sentence_percent, \
    feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity


def into_tokens(dataset, text):
    tokens = word_tokenize(text)

    bigrams = list(nltk.bigrams(tokens))
    new_tokens = []
    if dataset == 'Books':
        for i in range(len(bigrams)):
            if bigrams[i] == ('characters', 'development'):
                new_tokens.append('characters development')
            elif bigrams[i] == ('writing', 'style'):
                new_tokens.append('writing style')
            elif bigrams[i] == ('entertainment', 'value'):
                new_tokens.append('entertainment value')
            elif bigrams[i] == ('series', 'continuity'):
                new_tokens.append('series continuity')
            else:
                new_tokens.extend(bigrams[i])
    elif dataset == 'movie_and_tv':
        for i in range(len(bigrams)):
            if bigrams[i] == ('entertainment', 'value'):
                new_tokens.append('entertainment value')
            else:
                new_tokens.extend(bigrams[i])
    else:
        for i in range(len(bigrams)):
            if bigrams[i] == ('value', 'for', 'money'):
                new_tokens.append('value for money')
            elif bigrams[i] == ('room', 'size'):
                new_tokens.append('room size')
            else:
                new_tokens.extend(bigrams[i])
    return new_tokens


dataset_name = "movie_and_tv"  # movie_and_tv TripAdvisor Books

if dataset_name == 'Books':
    path = f'data/nrt_result/{dataset_name}/analysis.json'
    save_path = f'data/nrt_result/{dataset_name}/result.json'
elif dataset_name == 'movie_and_tv':
    path = f'data/nrt_result/{dataset_name}/amazon/analysis.json'
    save_path = f'data/nrt_result/{dataset_name}/amazon/result.json'
else:
    path = f'data/nrt_result/{dataset_name}/amazon/analysis.json'
    save_path = f'data/nrt_result/{dataset_name}/amazon/result.json'

gt_list = []
predict_list = []
feature_set = set()

if dataset_name == 'Books':
    aspect_list = ['Genre', 'Themes', 'Setting', 'Plot', 'Pacing', 'Characters Development', 'Writing Style',
                   'Entertainment Value', 'Series Continuity']
elif dataset_name == 'movie_and_tv':
    aspect_list = ['Plot', 'Acting', 'Cinematography', 'Music', 'Themes', 'Genre', 'Casting',
                   'Atmosphere', 'Pacing', 'Entertainment Value', 'Direction', 'Originality']
else:
    aspect_list = ['Cleanliness', 'Comfort', 'Location', 'Service', 'Value for Money', 'Room Size', 'Amenities',
                   'Food', 'Atmosphere']

for a in aspect_list:
    feature_set.add(a.lower())

with open(path, 'r') as f:
    data = json.load(f)
    for item in data:
        gt = item['ground-truth']
        predict = item['explanation']
        gt_tokens = into_tokens(dataset_name, gt)
        predict_tokens = into_tokens(dataset_name, predict)
        gt_list.append(gt_tokens)
        predict_list.append(predict_tokens)

BLEU1 = bleu_score(gt_list, predict_list, n_gram=1, smooth=False)
print('BLEU-1 {:7.4f}'.format(BLEU1))
BLEU4 = bleu_score(gt_list, predict_list, n_gram=4, smooth=False)
print('BLEU-4 {:7.4f}'.format(BLEU4))
USR, USN = unique_sentence_percent(predict_list)
print('USR {:7.4f} | USN {:7}'.format(USR, USN))
feature_batch = feature_detect(predict_list, feature_set)
DIV = feature_diversity(feature_batch)  # time-consuming
print('DIV {:7.4f}'.format(DIV))
FCR = feature_coverage_ratio(feature_batch, feature_set)
print('FCR {:7.4f}'.format(FCR))
FMR = feature_matching_ratio(feature_batch, feature_set)
print('FMR {:7.4f}'.format(FMR))
text_test = [' '.join(tokens) for tokens in gt_list]
text_predict = [' '.join(tokens) for tokens in predict_list]
ROUGE = rouge_score(text_test, text_predict)  # a dictionary
for (k, v) in ROUGE.items():
    print('{} {:7.4f}'.format(k, v))

result = {}
result['BLEU1'] = 'BLEU-1 {:7.4f}'.format(BLEU1)
result['BLEU4'] = 'BLEU-4 {:7.4f}'.format(BLEU4)
for (k, v) in ROUGE.items():
    result[k] = '{} {:7.4f}'.format(k, v)

with open(save_path, 'w') as f:
    json.dump(result, f, indent=4)
