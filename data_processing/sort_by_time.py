import json

dataset_name = 'Books'
rating_index = 'rating' if dataset_name == 'image_review_all' else 'overall'
id_index = 'user_id' if dataset_name == 'image_review_all' else 'reviewerID'
asin_index = 'business_id' if dataset_name == 'image_review_all' else 'asin'
txt_index = 'review_text' if dataset_name == 'image_review_all' else 'reviewText'

data_after_path = f'data/data_after_sparsity_processing/{dataset_name}/dataAfter_{dataset_name}.json'
hist_path = f'data/data_after_sparsity_processing/{dataset_name}/userHistory_{dataset_name}.json'
user_prompt = f'data/data_sort_by_time/{dataset_name}/user_prompt_{dataset_name}.json'
item_prompt = f'data/data_sort_by_time/{dataset_name}/item_prompt_{dataset_name}.json'


def Wash(data_after_path, hist_path, user_prompt, item_prompt):
    # 定义函数获取最终推荐项目，用户评论（除去推荐项目，并且以需要推荐用户为中心），项目被评论（除去交互用户，并且以需要推荐项目为中心）
    with open(hist_path, 'r') as f:
        hist = json.load(f)
    with open(data_after_path, 'r') as f:
        data = json.load(f)

    user_dirt = {}
    item_dirt = {}
    # 对历史记录按照时间排序降序，并且将最后一个大于平均评分的项目作为推荐项目
    # user2924
    for k, v in hist.items():
        user_list = []
        for df in data:
            if (df[id_index] == k):
                user_list.append(df)
        user_list = sorted(user_list, reverse=True, key=lambda x: x["unixReviewTime"])
        user_dirt[k] = user_list

    for item in data:
        iid = item[asin_index]
        if iid not in item_dirt:
            item_dirt[iid] = [item]
        else:
            item_dirt[iid].append(item)

    for item in item_dirt.items():
        iid = item[0]
        list = item[1]
        sorted_list = sorted(list, reverse=True, key=lambda x: x["unixReviewTime"])
        item_dirt[iid] = sorted_list

    with open(user_prompt, 'w') as json_file:
        json.dump(user_dirt, json_file)
    with open(item_prompt, 'w') as json_file:
        json.dump(item_dirt, json_file)


Wash(data_after_path, hist_path, user_prompt, item_prompt)

'''
to_dict(data_after_path, dict_save_path)
wash(dict_save_path, hist_path, hist_wash_by_rating_path)
with open(hist_wash_by_rating_path, 'r') as f:
    hist = json.load(f)
with open(dict_save_path, 'r') as f:
    dict = json.load(f)
num = 0
for k, v in hist.items():
    key = k + "_" + v[-1]
    if dict[key]['rating'] <= 3:
        num += 1
print(num)

{"overall": 1.0, 
"vote": "6", 
"verified": false, 
"reviewTime": "08 4, 2008", 
"reviewerID": "A3TE383URF1TBW", 
"asin": "B000RZPW9W", 
"style": {"Edition:": " Standard"}, 
"reviewerName": "Dirk Diggler", 
"reviewText": "This is the worst MMO I've ever played, period. I don't care how \"early\" this game 
is it was pushed out the door while it was still an alpha product and now you have the privilege of paying to test it for Funcom.\n\n
Class balance is a joke, end-game content consists of finding another player to 1 or 2-shot (kill in 1 or 2 hits,) then 
getting 1 or 2 shotted yourself and doing it all over again. PvP in this game consists of sitting at a spawn point and
 ganking other players before they can heal up.\n\nMassive PvP is a joke, keep sieges are a 5fps slideshow with over 1000ms
  ping times when all 96 players are on screen. It's not actually possible to win a siege due to bugs.\n\n
  Endgame raiding is also broken, it's not possible to beat a number of the bosses unless you exploit them.\n\n
  Crafting doesn't work - the quests are endlessly bugged as are the recipes.\n\nDid I mention the memory leaks and frequent
   client crashes? How about some nice broken quests? The half-working auction house system? A mail system that seems be
    designed to piss you off (seriously, you'll understand if you ever have to use it.)\n\nIn short, the game isn't done 
    - and it probably won't be done for another year or so. The parts that are done are either intentionally designed to frustrate you,
     or they were designed and coded at 8:45am before a 9:00am patch - they're not well thought out at all.\n\nStay far 
     away, there are _many_ better options out there.",
      "summary": "A trainwreck in slow motion", 
      "unixReviewTime": 1217808000}

交互历史文件
    "userID":[item1]
'''
