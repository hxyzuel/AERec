import json
import os

import matplotlib.pyplot as plt
import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    weight_file

    losses = []
    for epoch in range(world.TRAIN_epochs):
        if world.config['use_text']:
            bpr = utils.BPRLoss_text(Recmodel, world.config)
            dataset_name = world.config['dataset_name']
            item_potrait_path = f'..//data/{dataset_name}/item_embed_list.json'
            user_potrait_path = f'..//data/{dataset_name}/user_embed_list.json'
            with open(item_potrait_path, 'r') as f:
                item_potrait_list = json.load(f)
            with open(user_potrait_path, 'r') as f:
                user_potrait_list = json.load(f)
            item_potrait_tensor = torch.tensor(item_potrait_list).to(world.device)
            user_potrait_tensor = torch.tensor(user_potrait_list).to(world.device)

            start = time.time()
            if epoch % 10 == 0:
                cprint("[TEST]")
                Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            output_information, loss = Procedure.BPR_train_original_text(dataset, Recmodel, bpr, epoch,
                                                                         item_potrait_tensor,
                                                                         user_potrait_tensor, neg_k=Neg_k, w=w)
            losses.append(loss)
            print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')
            torch.save(Recmodel.state_dict(), weight_file)
        else:
            bpr = utils.BPRLoss(Recmodel, world.config)
            start = time.time()
            if epoch % 10 == 0:
                cprint("[TEST]")
                Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            output_information, loss = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
            losses.append(loss)
            print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')
            torch.save(Recmodel.state_dict(), weight_file)

    cprint("[FINAL TEST]")
    result = Procedure.Test(dataset, Recmodel, world.TRAIN_epochs - 1, w, world.config['multicore'])

    path_prefix = f"lgn-{world.dataset}-{world.config['use_text']}-seed_{world.config['seed']}-weight_{world.config['infonceloss_weight']}-temp_{world.config['nceloss_temperature']}-decay_{world.config['decay']}-lr_{world.config['lr']}-layer_{world.config['lightGCN_n_layers']}-dim_{world.config['latent_dim_rec']}-epoch_{world.config['epochs']}-batch_{world.config['bpr_batch_size']}"
    os.makedirs(path_prefix, exist_ok=True)

    path = path_prefix + '/losses.png'
    x = []
    for i in range(len(losses)):
        x.append(i)
    plt.figure()
    plt.plot(x, losses)
    plt.title('avg_losses')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(path)

    result_dump={}
    result_dump['ndcg']=result['ndcg'].tolist()
    result_dump['recall'] = result['recall'].tolist()
    with open(path_prefix + '/result.json', 'w') as f:
        json.dump(result_dump, f, indent=4)

finally:
    if world.tensorboard:
        w.close()
