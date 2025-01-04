import subprocess

assignment_update_frequency = 'every-2-epochs'
num_layers = 5
l2_penalty_factor = 5
lr1 = 1e-3
num_composition_centroid = 5
num_clusters = 300
anchor_weight = 0.5
epochs = 1000
dataset_name = 'video_games'

for lr in [1e-4]:
    subprocess.run([
        'python', 'engine.py', f'--dataset_name={dataset_name}',
        f'--assignment_update_frequency={assignment_update_frequency}'
    ])
# for lr in [1e-4, 1e-2]:
#     subprocess.run([
#         'python', 'engine.py', f'--dataset_name={dataset_name}', f'--num_layers={num_layers}',
#         f'--l2_penalty_factor={l2_penalty_factor}',
#         f'--lr={lr}', f'--num_composition_centroid={num_composition_centroid}', f'--num_clusters={num_clusters}',
#         f'--anchor_weight={anchor_weight}', f'--epochs={epochs}',
#         f'--assignment_update_frequency={assignment_update_frequency}'
#     ])

# dataset_name = 'Amazon/MoviesAndTV'
# log_name='amazon_moviesandtv'
# with open(f'{log_name}.log', 'w') as log_file:
#     subprocess.run([
#     'python', '-u', 'main.py', f'--data_path=../{dataset_name}/reviews.pickle', f'--index_dir=../{dataset_name}/1/',
#     '--cuda', f'--checkpoint=./{dataset_name}/'], stdout=log_file, stderr=subprocess.STDOUT)
