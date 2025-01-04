import subprocess

dataset_name = 'Books'
Ks = '[5, 10, 20, 50]'
embed_size = 128
layer_size = '[64, 64, 64]'
node_dropout = '[0.1]'
mess_dropout = '[0.1, 0.1, 0.1]'
save_flag = 1
epoch = 500
use_text = False

batch_size = 512  # 256,512,1024,2048
regs = '[1e-3]'  # '[1e-1],[1e-2],[1e-3],[1e-4],[1e-5]'
lr = 1e-3  # 1e-1,1e-2,1e-3,1e-4,1e-5
nceloss_temperature = 0.5  # 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9
infonceloss_weight = 1e-2  # 5e-2,5e-3,1e-3,5e-4

subprocess.run([
    'python', 'main.py', f'--dataset={dataset_name}',
    f'--regs={regs}',
    f'--embed_size={embed_size}',
    f'--layer_size={layer_size}',
    f'--lr={lr}',
    f'--save_flag={save_flag}',
    f'--batch_size={batch_size}',
    f'--epoch={epoch}',
    f'--node_dropout={node_dropout}',
    f'--mess_dropout={mess_dropout}',
    f'--Ks={Ks}',
    f'--use_text={use_text}',
    f'--nceloss_temperature={nceloss_temperature}',
    f'--infonceloss_weight={infonceloss_weight}'
])
