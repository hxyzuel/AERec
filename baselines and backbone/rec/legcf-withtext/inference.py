import os
import torch.optim
from data_loading.data_loader_for_inference import DataLoader_Inference
from utils.util import setup_seed, setup_recsys, print_opt, print_text, save_model, load_model
from set_parse import config_parser
from tensorboardX import SummaryWriter
from timeit import default_timer as timer
from datetime import timedelta
from funcs import train_model, test_model
import numpy as np
import scipy.sparse as sp
from parse import parse
from funcs import model_inference
import json

parser = config_parser()
opt = vars(parser.parse_args())
setup_seed(opt["seed"])
if torch.cuda.is_available():
    if not isinstance(opt['device_id'], int) and opt["device_id"].isnumeric():
        opt["device_id"] = f"cuda:{opt['device_id']}"
else:
    opt["device_id"] = "cpu"

opt["data_path"] = f"./data/{opt['dataset_name']}"

data_loader = DataLoader_Inference(opt=opt)
interact_mat = data_loader.interact_mat
opt["field_dims"] = list(interact_mat.shape)

opt["interact_mat"] = interact_mat

if torch.cuda.is_available():
    torch.cuda.empty_cache()

model_path = ('./res/video_games_assignment_update_frequency:every-2-epochs_num_cluster:300_seed'
              ':2020_num_composition_embs:5_lr:1e-03num_layers:5_num_clusters:300_l2_penalty:5.0_anchor_weight:0.5'
              '/best_checkpoint_before_assignment_update.pt')

recsys = load_model(opt, model_path)
result = model_inference(train_all_pos=data_loader.train_all_pos,test_all_pos=data_loader.test_all_pos,
                         recsys=recsys, opt=opt)
with open('./res/results.json', 'w') as file:
    json_data = json.dumps(result, indent=4)
    file.write(json_data)
