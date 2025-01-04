import subprocess
import torch
dataset_name='video_games'

subprocess.run([
        'python', 'inference.py', f'--dataset_name={dataset_name}'
    ])

# items_map={}
# users_map={}
# with open('./item_list.txt', 'r') as item_data:
#     lines = item_data.readlines()
#     for line in lines[1:]:
#         ids = line.split(' ')
#         items_map[ids[1].rstrip("\n")] = ids[0]
# with open('./user_list.txt', 'r') as user_data:
#     lines = user_data.readlines()
#     for line in lines[1:]:
#         ids = line.split(' ')
#         users_map[ids[1].rstrip("\n")] = ids[0]
#
# print(items_map['6'])
# tensor = torch.tensor([[5]])
# print(tensor.item())
# print(str(tensor.item()))