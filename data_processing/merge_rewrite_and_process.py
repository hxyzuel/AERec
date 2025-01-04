import json
import re


def post_process(result, aspect_list):
    if '>>' not in result:
        result = '>>'

    sub = '>>'
    index = result.find(sub)
    if index != -1:
        result = result[index:]

    sub2 = 'Note'
    index2 = result.find(sub2)
    if index2 != -1:
        result = result[:index2]

    if result[-1] != '.':
        index3 = result.rfind(sub)
        if index3 != -1:
            result = result[:index3]

    result.replace('\n\n', '\n')
    temp = result.split('\n')
    new = []
    sep = '\n'
    for sent in temp:
        sub_aspect = "Aspect"
        sub_maohao = ':'
        index_aspect = sent.find(sub_aspect)
        index_maohao = sent.find(sub_maohao)
        if index_aspect != -1:
            sent = sent[:index_aspect] + sent[index_maohao + 1:]

        # aspect_in_flag = False
        # for aspect in aspect_list:
        #     if aspect in sent:
        #         aspect_in_flag = True
        #
        # if not aspect_in_flag:
        #     continue

        if (sent == '' or 'Not applicable' in sent or 'not applicable' in sent or 'No specific' in sent
                or 'no specific' in sent or 'Not mentioned' in sent or 'not mentioned' in sent or 'No mention' in sent
                or 'no mention' in sent or 'Not mention' in sent or 'not mention' in sent or 'no inferences' in sent
                or 'No inferences' in sent or 'Not enough' in sent or 'not enough' in sent):
            continue

        pattern = r'no\s+(\w*)\s+mention'
        match = re.search(pattern, sent)
        if match:
            content_between = match.group(1)
            if content_between:
                continue

        pattern = r'No\s+(\w*)\s+mention'
        match = re.search(pattern, sent)
        if match:
            content_between = match.group(1)
            if content_between:
                continue

        pattern = r'not\s+(\w*)\s+enough'
        match = re.search(pattern, sent)
        if match:
            content_between = match.group(1)
            if content_between:
                continue

        pattern = r'Not\s+(\w*)\s+enough'
        match = re.search(pattern, sent)
        if match:
            content_between = match.group(1)
            if content_between:
                continue

        pattern = r'not\s+(\w*)\s+mentioned'
        match = re.search(pattern, sent)
        if match:
            content_between = match.group(1)
            if content_between:
                continue

        pattern = r'Not\s+(\w*)\s+mentioned'
        match = re.search(pattern, sent)
        if match:
            content_between = match.group(1)
            if content_between:
                continue

        pattern = r'not\s+(\w*)\s+mention'
        match = re.search(pattern, sent)
        if match:
            content_between = match.group(1)
            if content_between:
                continue

        pattern = r'Not\s+(\w*)\s+mention'
        match = re.search(pattern, sent)
        if match:
            content_between = match.group(1)
            if content_between:
                continue

        new.append(sent)

    result = sep.join(new)
    result.replace('\n\n', '\n')
    return result


dataset_name = 'Books'  # TripAdvisor movie_and_tv
path = f'data/rewrite_review/{dataset_name}'
aspect_path = f'data/aspects/{dataset_name}/aspects_full.txt'
save_path = f'data/rewrite_review/{dataset_name}/'
ns = 66 if dataset_name == 'TripAdvisor' else 38

aspects = []
with open(aspect_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        aspects.append(line.rstrip())

datas = {}
for n in range(0, ns + 1):
    print(n)
    temp_path = path + '/rewrite_review_' + str(n) + '.json'
    with open(temp_path, 'r') as f:
        data = json.load(f)
        for item in data.items():
            processed=post_process(item[1], aspects)
            if processed=='':
                continue
            datas[item[0]] = processed
with open(save_path + 'rewrite_review.json', 'w') as f:
    json.dump(datas, f, indent=4)
