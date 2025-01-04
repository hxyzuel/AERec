import json
from LLM import *
import math

split_num = 100
dataset_name = 'Books'
ground_truth_path1 = f'data/data_for_fine-tune/{dataset_name}/data_wo_ft.json'
ground_truth_path2 = f'data/data_sort_by_time/{dataset_name}/user_prompt_{dataset_name}.json'
aspect_path = f'data/aspects/{dataset_name}/aspects_full.txt'
save_path = f'data/rewrite_review/{dataset_name}/rewrite_review'

aspects = ''
with open(aspect_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        aspects += line.rstrip()
        aspects += ','

if dataset_name == 'movie_and_tv':
    prompt1 = f"""
    Assume you're an experienced reviewer of film and television works. \n\
    Based on a user's review,Tell me from which perspectives the user gave this review \n\
    Candidate Aspects:{aspects} \n\
    Examples of responses to each aspect are shown below:\n\
    >>Plot: The user seems to care about the plot and storytelling of the movie, criticizing some films for being too long or lacking in substance.\n\
    >>Acting: Opinions about the acting are divided, with some users praising James Brolin\'s performance and others finding it over-the-top or cheesy.\n\
    >>Cinematography: The user appreciates visually stunning films with breathtaking vistas, beautiful wardrobes, and excellent direction. They seem to enjoy films with a strong visual aesthetic.\n\
    >>Music: The user mentions the importance of music in setting the tone and atmosphere of a movie.\n\
    >>Themes: The reviews mention themes of politics, fascism, and the plight of the Jews, as well as the cultural implications of World War I and its aftermath.\n\
    >>Genre: The user seems to enjoy films from various genres, including animated specials, family films, and horror movies.\n\
    >>Atmosphere: The reviews suggest that the atmosphere is tense and suspenseful.\n\
    >>Pacing: The reviews suggest that the plot moves along quickly, with some critics noting that the pacing is fast-paced and action-packed.\n\
    >>Entertainment Value: The user seems to prioritize entertainment value, enjoying films that are fun and engaging.\n\
    >>Casting: The reviews praise the casting of Mary Steenburgen, Harry Dean Stanton, and Gary Basaraba, with some critics noting that the casting of Harry Dean Stanton as the guardian angel is questionable.\n\
    Ensures:\n\
    1.Format the response as:'>>Aspect_k:What the user say about the game in this aspect'.And response should NOT include anything else.(Aspect_k stands the k-th aspect)\n\
    2.If information of Aspect_M WAS NOT in the reviews, DO NOT response:DO NOT response:'>>Aspect_m:Aspect_m not mentioned' or '>>Aspect_m:Not enough information provided to infer Aspect_m.'.\n\
    3.If none of the above aspects are involved, please answer <None>\
    4.Aspects must select from Candidate Aspects:\n\
    5.Make reasonable inferences from the existing review.\n\
    6.Ensure inferences are logical, clear, and reasons are specific.\n\
    7.The response must be concise, within 120 words.\n\n\
    User Reviews:\n
    """
else:
    prompt1 = f"""
    Assume you're an experienced reviewer of works of literature. \n\
    Based on a user's review,Tell me from which perspectives the user gave this review \n\
    Candidate Aspects:{aspects} \n\
    >>Genre: The user enjoys crime novels \n\
    >>Themes: The user appreciates books that explore deep themes such as friendship, family, forgiveness, and justice. \n\
    >>Setting: The user enjoys stories set in historical contexts, particularly those that transport them to different time periods and places. \n\
    >>Plot: The user enjoys engaging and intriguing plots that involve mystery, tragedy, and emotional turmoil. \n\
    >>Pacing: The user prefers books that keep the reader engaged and excited, with intense and gripping moments. \n\
    >>Characters Development: The user values well-developed characters with depth and complexity, especially when they undergo personal growth or face moral dilemmas. \n\
    >>Writing Style: The user appreciates authors with strong writing skills, engaging dialogues, and insightful narratives. \n\
    >>Entertainment Value: The user enjoys books that provide entertainment through suspense, mystery, and engaging storytelling. \n\
    >>Series Continuity: The user enjoys series of books, as seen in their reviews where they express interest in continuing with the characters and stories in future installments. \n\
    \n\
    Ensures:\n\
    1.Format the response as:'>>Aspect_k:What the user say about the game in this aspect'.And response should NOT include anything else.(Aspect_k stands the k-th aspect)\n\
    2.If information of Aspect_M WAS NOT in the reviews, DO NOT response:DO NOT response:'>>Aspect_m:Aspect_m not mentioned' or '>>Aspect_m:Not enough information provided to infer Aspect_m.'.\n\
    3.If none of the above aspects are involved, please answer <None>\
    4.Aspects must select from Candidate Aspects:\n\
    5.Make reasonable inferences from the existing review.\n\
    6.Ensure inferences are logical, clear, and reasons are specific.\n\
    7.The response must be concise, within 120 words.\n\n\
    User Reviews:\n
    """


def load_data(path1, path2, split_num=100):
    with open(path1, 'r') as f:
        datas = json.load(f)
    with open(path2, 'r') as f:
        reivew_datas = json.load(f)
    keys = list(datas.keys())
    list1 = []
    for key in keys:
        uid = key
        iid = datas[key]
        temps = reivew_datas[uid]
        for item in temps:
            if item['asin'] == iid:
                list1.append(item['reviewText'])
    num_keys = len(keys)
    tail = num_keys % split_num
    n = math.floor(num_keys / split_num)
    return keys, list1, n, tail


def few_shot_ground_truth(review, prompt):
    prompt = prompt + review
    return prompt


def get_rewtite(review, gpt, prompt):
    prompt = few_shot_ground_truth(review, prompt)
    input = [{"role": "user",
              "content": prompt}]
    rewrite = gpt.get_completion(prompt=input)
    return rewrite


# def rewrite(prompt):
#     with open(ground_truth_path, 'r') as f:
#         truth = json.load(f)
#     data_list = {}
#     gpt = GPT3(temperature=0.9)
#     # 重写数据格式{"userID":"reWrite"}
#     temp = 0
#     for k, v in truth.items():
#         '''
#         if temp>20:break
#         temp+=1
#         '''
#         ID = k
#         review = v
#         rewrite = get_rewtite(review, gpt, prompt)
#         data_list[ID] = rewrite
#     with open(save_path, 'w') as json_file:
#         json.dump(data_list, json_file)

def rewrite(user_keys, user_list, n, tail):
    if current > n:
        print("current值非法")
        return
    path = save_path + f'_{current}.json'
    result = {}
    model = GPT3(0.9)

    if current == n:
        user_keys_part = user_keys[n * split_num:n * split_num + tail]
        user_list_part = user_list[n * split_num:n * split_num + tail]
        for k, review in zip(user_keys_part, user_list_part):
            result[str(k)] = get_rewtite(review, model, prompt1)
        with open(path, 'w') as file:
            json.dump(result, file, indent=4)
    else:
        user_keys_part = user_keys[current * split_num:(current + 1) * split_num]
        user_list_part = user_list[current * split_num:(current + 1) * split_num]
        for k, review in zip(user_keys_part, user_list_part):
            result[str(k)] = get_rewtite(review, model, prompt1)
        with open(path, 'w') as file:
            json.dump(result, file, indent=4)


keys, list1, n, tail = load_data(ground_truth_path1, ground_truth_path2, split_num=100)
print(n, tail)# 38 84
for current in range(n + 1):
    rewrite(keys, list1, n, tail)
