dataset_name = 'Books'
split_num = 15
model = 'gpt3.5'
now = 300
mode = ['item','user']  #

import json
import importlib.util
from code_jsjx.LLM import *
import re
import math


def load_data(path, split_num=500):
    with open(path, 'r') as f:
        datas = json.load(f)
    keys = list(datas.keys())
    hist_list = []
    for key in keys:
        current_data = datas[key]
        current_hist = []
        for item in current_data:
            reivew = item['reviewText']
            current_hist.append(reivew)
        hist_list.append(current_hist)
    num_keys = len(keys)
    tail = num_keys % split_num
    n = math.floor(num_keys / split_num)
    return keys, hist_list, n, tail


def post_process(result, aspect_list):
    if '>>' not in result:
        result = '>>'

    sub = '>>'
    index = result.find(sub)
    if index != -1:
        result = result[index:]

    sub2 = 'Note'
    index2 = result.find(sub2)
    if index != -1:
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

        aspect_in_flag = False
        for aspect in aspect_list:
            if aspect in sent:
                aspect_in_flag = True

        if not aspect_in_flag:
            continue

        if (sent == '' or 'Not applicable' in sent or 'not applicable' in sent or 'No specific' in sent
                or 'no specific' in sent or 'Not mentioned' in sent or 'not mentioned' in sent or 'No mention' in sent
                or 'no mention' in sent or 'Not mention' in sent or 'not mention' in sent or 'no inferences' in sent
                or 'No inferences' in sent or 'not enough' in sent or 'Not enough' in sent or 'N/A' in sent):
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


def generate_user_portrait(user_keys, user_hist_list, n, tail, aspect_list, exp_list, current, model, user_save_path,
                           split_num=500):
    if current > n:
        print("current值非法")
        return
    save_path = user_save_path + f'_{current}.json'
    result = {}

    if current == n:
        user_keys_part = user_keys[n * split_num:n * split_num + tail]
        user_hist_list_part = user_hist_list[n * split_num:n * split_num + tail]
        for k, hist in zip(user_keys_part, user_hist_list_part):
            # prompt = module.user_prompt_constrcut(hist, aspect_list, exp_list, model)
            prompt = module.user_prompt_constrcut(hist, aspect_list, exp_list, model)
            input = {}
            input['role'] = 'user'
            input['content'] = prompt
            model = GPT3(0.5)
            portrait = model.get_completion(prompt=[input])
            portrait = post_process(portrait, aspect_list)
            result[str(k)] = portrait
        with open(save_path, 'w') as file:
            json.dump(result, file, indent=4)
    else:
        user_keys_part = user_keys[current * split_num:(current + 1) * split_num]
        user_hist_list_part = user_hist_list[current * split_num:(current + 1) * split_num]
        for k, hist in zip(user_keys_part, user_hist_list_part):
            # prompt = module.user_prompt_constrcut(hist, aspect_list, exp_list, model)
            prompt = module.user_prompt_constrcut(hist, aspect_list, exp_list, model)
            input = {}
            input['role'] = 'user'
            input['content'] = prompt
            model = GPT3(0.5)
            portrait = model.get_completion(prompt=[input])
            portrait = post_process(portrait, aspect_list)
            result[str(k)] = portrait
        with open(save_path, 'w') as file:
            json.dump(result, file, indent=4)


def generate_item_portrait(item_keys, item_hist_list, n, tail, aspect_list, exp_list, current, model, item_save_path,
                           split_num=500):
    if current > n:
        print("current值非法")
        return

    save_path = item_save_path + f'_{current}.json'
    result = {}
    if current == n:
        item_keys_part = item_keys[n * split_num:n * split_num + tail]
        item_hist_list_part = item_hist_list[n * split_num:n * split_num + tail]
        for k, hist in zip(item_keys_part, item_hist_list_part):
            # prompt = module.item_prompt_constrcut(hist, aspect_list, exp_list, model)
            prompt = module.item_prompt_constrcut(hist, aspect_list, exp_list, model)
            input = {}
            input['role'] = 'user'
            input['content'] = prompt
            model = GPT3(0.5)
            portrait = model.get_completion(prompt=[input])
            portrait = post_process(portrait, aspect_list)
            result[str(k)] = portrait
        with open(save_path, 'w') as file:
            json.dump(result, file, indent=4)
    else:
        item_keys_part = item_keys[current * split_num:(current + 1) * split_num]
        item_hist_list_part = item_hist_list[current * split_num:(current + 1) * split_num]
        for k, hist in zip(item_keys_part, item_hist_list_part):
            # prompt = module.item_prompt_constrcut(hist, aspect_list, exp_list, model)
            prompt = module.item_prompt_constrcut(hist, aspect_list, exp_list, model)
            input = {}
            input['role'] = 'user'
            input['content'] = prompt
            model = GPT3(0.5)
            portrait = model.get_completion(prompt=[input])
            portrait = post_process(portrait, aspect_list)
            result[str(k)] = portrait
        with open(save_path, 'w') as file:
            json.dump(result, file, indent=4)


# 文件路径在运行时确定，例如通过用户输入或其他方式获取
file_path = f'data/prompt_for_portrait_generation/{dataset_name}/prompt.py'
# 指定模块名称
module_name = "prompt"
# 创建一个 ModuleSpec 对象，指定模块名称和文件路径
spec = importlib.util.spec_from_file_location(module_name, file_path)
# 使用 import_module 方法导入模块
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

exp_item_name = 'exp_item'
exp_user_name = 'exp_user'
exp_item = getattr(module, exp_item_name)
exp_user = getattr(module, exp_user_name)
# 现在可以使用导入的模块了

id_index = 'user_id' if dataset_name == 'image_review_all' else 'reviewerID'
asin_index = 'business_id' if dataset_name == 'image_review_all' else 'asin'
txt_index = 'review_text' if dataset_name == 'image_review_all' else 'reviewText'

user_path = f'data/data_for_portrait_generation/{dataset_name}/user_prompt_{dataset_name}.json'
item_path = f'data/data_for_portrait_generation/{dataset_name}/item_prompt_{dataset_name}.json'

user_aspects_list_path = f'data/aspects/{dataset_name}/aspects_for_user_portrait_generation.txt'
item_aspects_list_path = f'data/aspects/{dataset_name}/aspects_for_item_portrait_generation.txt'

user_save_path = f'data/aspects/{dataset_name}/user_portrait.json'
item_save_path = f'data/aspects/{dataset_name}/item_portrait.json'

user_hist = {}
item_hist = {}
with open(user_path, 'r') as f:
    user_hist_dict = json.load(f)
for item in user_hist_dict.items():
    temp = []
    uid = item[0]
    hists = item[1]
    for hist in hists:
        temp.append(hist[txt_index])
    user_hist[uid] = temp
with open(item_path, 'r') as f:
    item_hist_dict = json.load(f)
for item in item_hist_dict.items():
    temp = []
    iid = item[0]
    hists = item[1]
    for hist in hists:
        temp.append(hist[txt_index])
    item_hist[iid] = temp

user_aspects_list = []
item_aspects_list = []
with open(user_aspects_list_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        user_aspects_list.append(line.rstrip())
with open(item_aspects_list_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        item_aspects_list.append(line.rstrip())

user_result = {}
item_result = {}

for mode in mode:  # ['user', 'item']:
    path = user_path if mode == 'user' else item_path
    keys, hist_list, n, tail = load_data(path, split_num=split_num)
    print(n, tail)

    # recover_current = 0  # 值为0~n-1,分n次生成,只用改这个
    # user_recover(recover_current, f'user_portrait_{recover_current}.json', user_path, aspects, model)

    for current in range(now, now + 1):
        if mode == 'user':
            generate_user_portrait(current=current, user_keys=keys, user_hist_list=hist_list, exp_list=exp_user,
                                   aspect_list=user_aspects_list, n=n, tail=tail, user_save_path=user_save_path,
                                   model=model, split_num=split_num)
        else:
            generate_item_portrait(current=current, item_keys=keys, item_hist_list=hist_list, exp_list=exp_item,
                                   aspect_list=item_aspects_list, n=n, tail=tail, item_save_path=item_save_path,
                                   model=model, split_num=split_num)
#
# if model == 'llama3':
#     user_keys_list = list(user_hist.keys())
#     item_keys_list = list(item_hist.keys())
#
#     for iid in item_keys_list[2100:2121]:
#         reviews = item_hist[iid]
#         prompt = module.item_prompt_constrcut(reviews, item_aspects_list, exp_item, model)
#         input = {}
#         input['role'] = 'user'
#         input['content'] = prompt
#         result = llama3([input])
#         #result = post_process(result,item_aspects_list)
#
#         item_result[iid] = result
#
#     with open(item_save_path, 'w') as f:
#         json.dump(item_result, f, indent=4)
#
#     for uid in user_keys_list[2100:2121]:
#         reviews = user_hist[uid]
#         prompt = module.user_prompt_constrcut(reviews, user_aspects_list, exp_user, model)
#         input = {}
#         input['role'] = 'user'
#         input['content'] = prompt
#         result = llama3([input])
#
#         #result = post_process(result,user_aspects_list)
#
#         user_result[uid] = result
#
#     with open(user_save_path, 'w') as f:
#         json.dump(user_result, f, indent=4)
