import subprocess

dataset_name = 'Video_Games'
bpr_batch = 2048  # 256,512,1024
layer = 2  # 1,3,4,5
lr = 1e-3  # 定 5e-2,1e-2,5e-3,1e-3,5e-4
decay = 1e-3  # 1e-2,5e-3,5e-4,1e-4
# seed = 2021
epochs = 500  # 100,200,300,400
# nceloss_temperature = 0.2  # 定 0.1, 0.2, 0.3, 0.4,0.5,0.6, 0.7, 0.8, 0.9
infonceloss_weight = 1e-2  # 定 5e-2,1e-2,5e-3,1e-3,5e-4
use_text = True
use_text_user = True
use_text_item = True

for nceloss_temperature in [0.1]:
    for seed in [2022]:
        subprocess.run([
            'python', 'main.py', f'--dataset={dataset_name}',
            f'--layer={layer}',
            f'--lr={lr}',
            f'--decay={decay}',
            f'--seed={seed}',
            f'--epochs={epochs}',
            f'--bpr_batch={bpr_batch}',
            f'--use_text={use_text}',
            f'--nceloss_temperature={nceloss_temperature}',
            f'--infonceloss_weight={infonceloss_weight}',
            f'--use_text_user={use_text_user}',
            f'--use_text_item={use_text_item}'
        ])

# dataset_name = 'Video_Games'
# bpr_batch = 2048  # 256,512,1024
# layer = 2  #定 1,3,4,5
# lr = 1e-3  # 定 5e-2,1e-2,5e-3,1e-3,5e-4
# decay = 1e-3  # 1e-2,5e-3,5e-4,1e-4
# # seed = 2021
# epochs = 500  # 100,200,300,400
# nceloss_temperature = 0.2  # 定 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9
# infonceloss_weight = 1e-2  # 定 5e-2,5e-3,1e-3,5e-4
# use_text = True
# use_text_user=True
# use_text_item=True
#
# for layer in [1, 3, 4, 5]:
#     for seed in [2020, 2021, 2022]:
#         subprocess.run([
#             'python', 'main.py', f'--dataset={dataset_name}',
#             f'--layer={layer}',
#             f'--lr={lr}',
#             f'--decay={decay}',
#             f'--seed={seed}',
#             f'--epochs={epochs}',
#             f'--bpr_batch={bpr_batch}',
#             f'--use_text={use_text}',
#             f'--nceloss_temperature={nceloss_temperature}',
#             f'--infonceloss_weight={infonceloss_weight}',
#             f'--use_text_user={use_text_user}',
#             f'--use_text_item={use_text_item}'
#         ])
