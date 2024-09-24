import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from model.neg_worker import NegModels
from model.pos_worker import PosModels
from model.scan_worker import ScanModels
from model.edl_loss import EvidenceLoss
from model.manager_trust import ManagerModel2
from util.data_util import load_video_features, save_json, load_video_features_activity
from util.data_gen_glove import gen_or_load_dataset
from util.data_loader_pos_glove import get_train_loader, get_test_loader
from util.runner_utils_t7 import set_th_config
from util.rl_utils import get_relative_loc,calculate_reward_batch_withstop_start, calculate_reward_batch_withstop_end, renew_state, calculate_IoU, renew_state_test,calculate_RL_IoU_batch, adjust_lr, renew_state_scan,  calculate_reward_batch_withstop_start_scan, calculate_reward_batch_withstop_end_scan, renew_state_scan_test, get_ious
import random
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from model.text_encoding import WordEmbedding


parser = argparse.ArgumentParser()
# data parameters
parser.add_argument('--save_dir', type=str, default='datasets_t7', help='path to save processed dataset')
parser.add_argument('--task', type=str, default='tacos', help='target task')
parser.add_argument('--fv', type=str, default='new', help='[new | org] for visual features')
parser.add_argument('--max_pos_len', type=int, default=1024, help='maximal position sequence length allowed')
parser.add_argument('--num_steps', type=int, default=10, help='transformer layer number')
# model parameters
parser.add_argument("--word_dim", type=int, default=300, help="word embedding dimension")
parser.add_argument("--video_feature_dim", type=int, default=1024, help="video feature input dimension")
parser.add_argument('--n_rl', type=int, default=16, help='')
parser.add_argument("--n_word", type=int, default=10, help='charades=10, activitynet=25')
parser.add_argument("--v_layer", type=int, default=2)
parser.add_argument("--s_layer", type=int, default=2)
parser.add_argument("--v_hidden", type=int, default=256)
parser.add_argument("--s_hidden", type=int, default=256)
parser.add_argument("--l_cross", type=int, default=512)
parser.add_argument("--n_loc", type=int, default=128)
parser.add_argument("--n_inter", type=int, default=256)
parser.add_argument("--n_feature", type=int, default=1024)
parser.add_argument("--n_hidden", type=int, default=512)
parser.add_argument("--pos_n_action", type=int, default=7)
parser.add_argument("--neg_n_action", type=int, default=7)
parser.add_argument("--scan_n_action", type=int, default=7)
parser.add_argument("--drop_rate", type=float, default=0.5)
# training/evaluation parameters
parser.add_argument("--gpu_idx", type=str, default="0", help="GPU index")
parser.add_argument("--seed", type=int, default=2013, help="random seed")
parser.add_argument("--mode", type=str, default="train", help="[train | test]")
parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--num_train_steps", type=int, default=None, help="number of training steps")
parser.add_argument("--init_lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--warmup_init_lr", type=float, default=-1)
parser.add_argument("--warmup_updates", type=float, default=4000)
parser.add_argument("--clip_norm", type=float, default=1.0, help="gradient clip norm")
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="warmup proportion")
parser.add_argument("--period", type=int, default=100, help="training loss print period")
parser.add_argument('--model_dir', type=str, default='ckpt_t7', help='path to save trained model weights')
parser.add_argument('--model_name', type=str, default='agent3scanv2loc', help='model name')
parser.add_argument('--suffix', type=str, default=None, help='set to the last `_xxx` in ckpt repo to eval results')
parser.add_argument("--lambda_0", type=float, default=0.1)
parser.add_argument("--lambda_1", type=float, default=1.0)
parser.add_argument("--lambda_2", type=float, default=1.0)
parser.add_argument("--lambda_3", type=float, default=1.0)
parser.add_argument("--delta0", type=float, default=0.16)
parser.add_argument("--delta1", type=float, default=0.05)
parser.add_argument("--delta2", type=float, default=0.02)
parser.add_argument("--sd0", type=float, default=0.14)
parser.add_argument("--sd1", type=float, default=0.1)
parser.add_argument("--sd2", type=float, default=0.06)
parser.add_argument("--move", type=float, default=0.1)
parser.add_argument("--beta", type=float, default=-0.8)
parser.add_argument("--threshold", type=float, default=0.05)
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--scan_reward_value", type=float, default=0.000)
parser.add_argument("--stop_reward_value", type=float, default=-0.001)
parser.add_argument("--n_save", type=int, default=50)
configs = parser.parse_args()

# set tensorflow configs
set_th_config(configs.seed)
pos_start_point,pos_end_point = 1 / 4.0, 3 / 4.0
#neg_start_point,neg_end_point = 0.0, 1.0
neg_start_point,neg_end_point = 1 / 4.0, 3 / 4.0
scan_start_point, scan_end_point = 0.0, (0.0 + configs.sd0)
STOP_ACTION_SCAN = 7
print("Pos: (%f, %f);  Neg: (%f, %f)." % (pos_start_point,pos_end_point, neg_start_point,neg_end_point))
print((scan_start_point, scan_end_point))
# prepare or load dataset
dataset = gen_or_load_dataset(configs)
configs.char_size = dataset['n_chars']
configs.word_size = dataset['n_words']
# get train and test loader
# visual_features = load_video_features(os.path.join('data', 'features', configs.task, configs.fv), configs.max_pos_len)

if configs.task == 'charades':
    visual_features = load_video_features(os.path.join('../../Charades-STA',"charades_i3d"), configs.max_pos_len)
elif configs.task == 'activitynet':
    visual_features = load_video_features_activity(os.path.join('../../',"ActivityNet"), configs.max_pos_len)
else:
    raise ValueError('Unknown task {}!!!'.format(configs.task))
train_loader = get_train_loader(dataset=dataset['train_set'], video_features=visual_features, configs=configs)
val_loader = None if dataset['val_set'] is None else get_test_loader(dataset['val_set'], visual_features, configs)
test_loader = get_test_loader(dataset=dataset['test_set'], video_features=visual_features, configs=configs)
train_loader_2 = get_test_loader(dataset=dataset['train_set'], video_features=visual_features, configs=configs)
configs.num_train_steps = len(train_loader) * configs.epochs
num_train_batches = len(train_loader)
num_val_batches = 0 if val_loader is None else len(val_loader)
num_test_batches = len(test_loader)

# Device configuration
cuda_str = 'cuda' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx)
device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
configs.device = str(device)
print(configs)
# create model dir
home_dir = os.path.join(configs.model_dir, '_'.join([configs.model_name, configs.task, configs.fv,
                                                     str(configs.max_pos_len)]))
if configs.suffix is not None:
    home_dir = home_dir + '_' + configs.suffix
model_dir = os.path.join(home_dir, "model")


def new_test_model(model, ini_model, data_loader, is_pos):
    model.eval()
    ini_model.eval()
    positive_5 = 0
    positive_7 = 0
    count = 0
    #it_count = 0
    step = 0
    for data in data_loader:
        #it_count += 1
        #if it_count > 2:
        #    break
        loc, _, _, gt_loc,_ = run_model(model, ini_model, data, is_pos)
        curr_batch_size = len(data[0])
        for i in range(curr_batch_size):
            count += 1
            tIoU = calculate_IoU(gt_loc[i], loc[i])
            #print(tIoU)
            if float(tIoU) >= 0.5:
                positive_5 = positive_5 + 1
            if float(tIoU) >= 0.7:
                positive_7 = positive_7 + 1
    step = step + 1
    iou_5 = float(positive_5) / count * 100
    iou_7 = float(positive_7) / count * 100
    print('The accuray when tIoU is higher than 0.5 is %.2f' % (iou_5))
    print('The accuray when tIoU is higher than 0.7 is %.2f' % (iou_7))
    return iou_5, iou_7
def run_model(model, ini_model, data, is_pos):
    records, vfeats, vfeat_lens, w_lens, word_ids, char_ids, s_labels, e_labels, label = data
    curr_batch_size = vfeats.size(0)
    loc = torch.zeros(curr_batch_size, 2).float()
    gt_loc = (torch.stack((s_labels / vfeat_lens, e_labels / vfeat_lens),dim=1))

    start_h_t = torch.zeros(curr_batch_size, configs.n_hidden).float().to(device)
    end_h_t = torch.zeros(curr_batch_size, configs.n_hidden).float().to(device)
    Predict_IoUs = torch.zeros(configs.num_steps, curr_batch_size)
    Predict_dis = torch.zeros(configs.num_steps, curr_batch_size, 2).to(device)
    act_locations = torch.zeros(configs.num_steps, curr_batch_size, 2)
    evidences = torch.zeros(configs.num_steps, curr_batch_size, configs.n_rl)
    if is_pos:
        loc[:, 0] = pos_start_point * torch.ones_like(loc[:, 0])
        loc[:, 1] = pos_end_point * torch.ones_like(loc[:, 1])
    else:
        loc[:, 0] = neg_start_point * torch.ones_like(loc[:, 0])
        loc[:, 1] = neg_end_point * torch.ones_like(loc[:, 1])
    m_s_batch = torch.ones(curr_batch_size).float().to(device)
    m_e_batch = torch.ones(curr_batch_size).float().to(device)

    visual_fea = None
    sen_fea = None
    for i_temp in range(configs.num_steps):
        pre_location = loc.clone().to(device) 
        vfeats  = vfeats.to(device)
        w_lens, word_ids = w_lens.to(device), word_ids.to(device)
        sent = ini_model(word_ids)
        visual_fea, sen_fea, start_h_t, start_logit, start_v_t, end_h_t, \
                    end_logit, end_v_t, p_tIoU, p_loc, p_dis, relative_loc = model(i_temp, vfeats, \
                            sent, visual_fea, sen_fea, pre_location, \
                            w_lens, vfeat_lens,\
                            start_h_t, end_h_t)
        for i in range(curr_batch_size):
            if int(m_s_batch[i]) == 0 and int(m_e_batch[i]) == 0:
                continue
            start_prob = F.softmax(start_logit[i].unsqueeze(0), dim=1)
            start_action = start_prob.max(1, keepdim=True)[1].data.cpu().numpy()[0, 0]
            end_prob = F.softmax(end_logit[i].unsqueeze(0), dim=1)
            end_action = end_prob.max(1, keepdim=True)[1].data.cpu().numpy()[0, 0]
            if is_pos:
                new_loc = renew_state_test(loc[i].unsqueeze(0), start_action, end_action, \
                                configs.delta0, configs.delta1, configs.delta2, int(m_s_batch[i]), int(m_e_batch[i]))
            else:
                new_loc = renew_state_test(loc[i].unsqueeze(0), start_action, end_action, \
                                configs.delta0, configs.delta1, configs.delta2, int(m_s_batch[i]), int(m_e_batch[i]))
            loc[i,:] = new_loc.squeeze(0)
            if start_action == 6:
                m_s_batch[i] = 0
            if end_action == 6:
                m_e_batch[i] = 0
        evidences[i_temp, :] = relative_loc
        Predict_IoUs[i_temp, :] = p_tIoU.squeeze(1)
        Predict_dis[i_temp, :, :] = p_dis
        act_locations[i_temp, :, :] = loc

    return loc, evidences, Predict_IoUs, gt_loc, act_locations


def scan_test_model(model, ini_model, data_loader):
    model.eval()
    ini_model.eval()
    positive_5 = 0
    positive_7 = 0
    count = 0
    #it_count = 0
    step = 0
    for data in data_loader:
        #it_count += 1
        #if it_count > 2:
        #    break
        scan_loc, accept_loc, _, _, gt_loc,_ = run_scan_model(model, ini_model, data)
        curr_batch_size = len(data[0])
        for i in range(curr_batch_size):
            count += 1
            tIoU = calculate_IoU(gt_loc[i], accept_loc[i])
            #print(tIoU)
            if float(tIoU) >= 0.5:
                positive_5 = positive_5 + 1
            if float(tIoU) >= 0.7:
                positive_7 = positive_7 + 1
    step = step + 1
    iou_5 = float(positive_5) / count * 100
    iou_7 = float(positive_7) / count * 100
    print('The accuray when tIoU is higher than 0.5 is %.2f' % (iou_5))
    print('The accuray when tIoU is higher than 0.7 is %.2f' % (iou_7))
    return iou_5, iou_7

def run_scan_model(model, ini_model, data):
    records, vfeats, vfeat_lens, w_lens, word_ids, char_ids, s_labels, e_labels, label = data
    curr_batch_size = vfeats.size(0)
    gt_loc = (torch.stack((s_labels / vfeat_lens, e_labels / vfeat_lens),dim=1))

    start_h_t = torch.zeros(curr_batch_size, configs.n_hidden).float().to(device)
    end_h_t = torch.zeros(curr_batch_size, configs.n_hidden).float().to(device)
    Predict_IoUs = torch.zeros(configs.num_steps, curr_batch_size)
    Predict_dis = torch.zeros(configs.num_steps, curr_batch_size, 2).to(device)
    act_locations = torch.zeros(configs.num_steps, curr_batch_size, 2)
    evidences = torch.zeros(configs.num_steps, curr_batch_size, configs.n_rl)
    scan_loc = torch.zeros(curr_batch_size, 2).float()
    scan_loc[:, 0] = scan_start_point * torch.ones_like(scan_loc[:, 0])
    scan_loc[:, 1] = scan_end_point * torch.ones_like(scan_loc[:, 1])
    accept_loc = scan_loc.clone()
    m_s_batch = torch.ones(curr_batch_size).float().to(device)
    m_e_batch = torch.ones(curr_batch_size).float().to(device)

    visual_fea = None
    sen_fea = None
    for i_temp in range(configs.num_steps):
        pre_scan_location = scan_loc.clone().to(device)  
        vfeats  = vfeats.to(device)
        w_lens, word_ids = w_lens.to(device), word_ids.to(device)
        sent = ini_model(word_ids)
        visual_fea, sen_fea, start_h_t, start_logit, start_v_t, end_h_t, \
                    end_logit, end_v_t, p_tIoU, p_loc, p_dis, relative_loc = model(i_temp, vfeats, \
                            sent, visual_fea, sen_fea, pre_scan_location, \
                            w_lens, vfeat_lens,\
                            start_h_t, end_h_t)
        for i in range(curr_batch_size):
            if int(m_s_batch[i]) == 0 and int(m_e_batch[i]) == 0:
                continue
            start_prob = F.softmax(start_logit[i].unsqueeze(0), dim=1)
            start_action = start_prob.max(1, keepdim=True)[1].data.cpu().numpy()[0, 0]
            end_prob = F.softmax(end_logit[i].unsqueeze(0), dim=1)
            end_action = end_prob.max(1, keepdim=True)[1].data.cpu().numpy()[0, 0]
            new_scan_loc, new_accept_loc = renew_state_scan_test(scan_loc[i].unsqueeze(0), accept_loc[i].unsqueeze(0), \
                                                                 start_action, end_action, configs.sd0, configs.sd1, configs.sd2, configs.move, int(m_s_batch[i]), int(m_e_batch[i]))
            '''
            if i == 1:
                print("Step: %i, Action: %i" % (i_temp, int(start_action)))
                print("Step: %i, Action: %i" % (i_temp, int(end_action)))
                print("old_scan_loc:(%f, %f): old_accept_loc(%f, %f)" % ( scan_loc[i][0], scan_loc[i][1], accept_loc[i][0], accept_loc[i][1]))
                print("gt_loc:(%f, %f), new_scan_loc:(%f, %f): new_accept_loc(%f, %f)" % (gt_loc[i][0], gt_loc[i][1], new_scan_loc[0][0], new_scan_loc[0][1], new_accept_loc[0][0], new_accept_loc[0][1]))
                print(new_scan_loc.squeeze(0))
                print(scan_loc[i])
            '''
            scan_loc[i,:] = new_scan_loc.squeeze(0)
            accept_loc[i,:] = new_accept_loc.squeeze(0)
            if start_action == STOP_ACTION_SCAN:
                m_s_batch[i] = 0
            if end_action == STOP_ACTION_SCAN:
                m_e_batch[i] = 0
        evidences[i_temp, :] = relative_loc
        Predict_IoUs[i_temp, :] = p_tIoU.squeeze(1)
        Predict_dis[i_temp, :, :] = p_dis
        act_locations[i_temp, :, :] = new_scan_loc

    return scan_loc, accept_loc, evidences, Predict_IoUs, gt_loc, act_locations




def m_selected_model(pos_model, neg_model, scan_model, m_model, ini_model, data_loader):
    pos_model.eval()
    neg_model.eval()
    scan_model.eval()
    m_model.eval()
    ini_model.eval()
    positive_5 = 0
    positive_7 = 0
    count = 0
    #it_count = 0
    step = 0
    all_golden = []
    all_predict = []
    with torch.no_grad():
        for data in data_loader:
            pos_loc, pos_evi, pos_iou, gt_loc, pos_loc_save = run_model(pos_model, ini_model, data, True)
            neg_loc, neg_evi, neg_iou, gt_loc, neg_loc_save = run_model(neg_model, ini_model, data, False)
            _, accept_loc, scan_evi, scan_iou, gt_loc, scan_loc_save = run_scan_model(scan_model, ini_model, data)
            pos_data = (pos_iou.to(device), pos_evi.to(device), pos_loc_save.to(device))
            neg_data = (neg_iou.to(device), neg_evi.to(device), neg_loc_save.to(device))
            scan_data = (scan_iou.to(device), scan_evi.to(device), scan_loc_save.to(device))
            pos_logit = man_model(pos_data)
            neg_logit = man_model(neg_data)
            scan_logit = man_model(scan_data)
            choice_label = torch.argmax(torch.stack([pos_logit, neg_logit, scan_logit], dim=1), dim=1)
            pos_tiou, neg_tiou, scan_tiou = get_ious(gt_loc, pos_loc, neg_loc, accept_loc)
            pos_tiou, neg_tiou, scan_tiou = pos_tiou.to(device), neg_tiou.to(device), scan_tiou.to(device)
            select_label = torch.argmax(torch.stack([pos_tiou, neg_tiou, scan_tiou], dim=1), dim=1)
            all_golden += select_label.tolist()
            all_predict += choice_label.tolist()
            curr_batch_size = pos_iou.size(1)
            for i in range(curr_batch_size):
                count += 1
                if choice_label[i] == 0:
                    tIoU = calculate_IoU(gt_loc[i], pos_loc[i])
                elif choice_label[i] == 1:
                    tIoU = calculate_IoU(gt_loc[i], neg_loc[i])
                else:
                    tIoU = calculate_IoU(gt_loc[i], accept_loc[i])
                #print(tIoU)
                if float(tIoU) >= 0.5:
                    positive_5 = positive_5 + 1
                if float(tIoU) >= 0.7:
                    positive_7 = positive_7 + 1
    step = step + 1
    iou_5 = float(positive_5) / count * 100
    iou_7 = float(positive_7) / count * 100
    print('The accuray when tIoU is higher than 0.5 is %.2f' % (iou_5))
    print('The accuray when tIoU is higher than 0.7 is %.2f' % (iou_7))
    print('Select Acc: %f' % (accuracy_score(all_golden, all_predict)))
    return iou_5, iou_7

def m_selected_model_new(pos_model, neg_model, scan_model, m_model, ini_model, data_loader):
    pos_model.eval()
    neg_model.eval()
    scan_model.eval()
    m_model.eval()
    ini_model.eval()
    positive_5 = 0
    positive_7 = 0
    pos_5, neg_5, scan_5 = 0,0,0
    pos_7, neg_7, scan_7 = 0,0,0
    count = 0
    #it_count = 0
    step = 0
    all_golden = []
    all_predict = []
    with torch.no_grad():
        for data in data_loader:
            pos_loc, pos_evi, pos_iou, gt_loc, pos_loc_save = run_model(pos_model, ini_model, data, True)
            neg_loc, neg_evi, neg_iou, gt_loc, neg_loc_save = run_model(neg_model, ini_model, data, False)
            _, accept_loc, scan_evi, scan_iou, gt_loc, scan_loc_save = run_scan_model(scan_model, ini_model, data)
            pos_data = (pos_iou.to(device), pos_evi.to(device), pos_loc_save.to(device))
            neg_data = (neg_iou.to(device), neg_evi.to(device), neg_loc_save.to(device))
            scan_data = (scan_iou.to(device), scan_evi.to(device), scan_loc_save.to(device))
            pos_logit = man_model(pos_data, pos_loc.detach().to(device))
            neg_logit = man_model(neg_data, neg_loc.detach().to(device))
            scan_logit = man_model(scan_data, accept_loc.detach().to(device))
            choice_label = torch.argmax(torch.stack([pos_logit, neg_logit, scan_logit], dim=1), dim=1).tolist()
            pos_tiou, neg_tiou, scan_tiou = get_ious(gt_loc, pos_loc, neg_loc, accept_loc)
            pos_tiou, neg_tiou, scan_tiou = pos_tiou.to(device), neg_tiou.to(device), scan_tiou.to(device)
            select_label = torch.argmax(torch.stack([pos_tiou, neg_tiou, scan_tiou], dim=1), dim=1)
            all_golden += select_label.tolist()
            all_predict += choice_label
            curr_batch_size = pos_iou.size(1)
            for i in range(curr_batch_size):
                count += 1
                pos_tIoU, neg_tIoU, scan_tIoU = calculate_IoU(gt_loc[i], pos_loc[i]), calculate_IoU(gt_loc[i], neg_loc[i]), calculate_IoU(gt_loc[i], accept_loc[i])
                if choice_label[i] == 0:
                    tIoU = pos_tIoU
                elif choice_label[i] == 1:
                    tIoU = neg_tIoU
                else:
                    tIoU = scan_tIoU
                #print(tIoU)
                if float(tIoU) >= 0.5:
                    positive_5 = positive_5 + 1
                if float(tIoU) >= 0.7:
                    positive_7 = positive_7 + 1
                if float(pos_tIoU) >= 0.5:
                    pos_5 = pos_5 + 1
                if float(pos_tIoU) >= 0.7:
                    pos_7 = pos_7 + 1
                if float(neg_tIoU) >= 0.5:
                    neg_5 = neg_5 + 1
                if float(neg_tIoU) >= 0.7:
                    neg_7 = neg_7 + 1
                if float(scan_tIoU) >= 0.5:
                    scan_5 = scan_5 + 1
                if float(scan_tIoU) >= 0.7:
                    scan_7 = scan_7 + 1
    step = step + 1
    iou_5 = float(positive_5) / count * 100
    iou_7 = float(positive_7) / count * 100
    pos_iou_5 = float(pos_5) / count * 100
    pos_iou_7 = float(pos_7) / count * 100
    neg_iou_5 = float(neg_5) / count * 100
    neg_iou_7 = float(neg_7) / count * 100
    scan_iou_5 = float(scan_5) / count * 100
    scan_iou_7 = float(scan_7) / count * 100
    print("Select")
    print('The accuray when tIoU is higher than 0.5 is %.2f' % (iou_5))
    print('The accuray when tIoU is higher than 0.7 is %.2f' % (iou_7))
    print('Select Acc: %f' % (accuracy_score(all_golden, all_predict)))
    print("Pos")
    print('The accuray when tIoU is higher than 0.5 is %.2f' % (pos_iou_5))
    print('The accuray when tIoU is higher than 0.7 is %.2f' % (pos_iou_7))
    print("Neg")
    print('The accuray when tIoU is higher than 0.5 is %.2f' % (neg_iou_5))
    print('The accuray when tIoU is higher than 0.7 is %.2f' % (neg_iou_7))
    print("Scan")
    print('The accuray when tIoU is higher than 0.5 is %.2f' % (scan_iou_5))
    print('The accuray when tIoU is higher than 0.7 is %.2f' % (scan_iou_7))
    return iou_5, iou_7

def get_loss(model, ini_model, data, is_pos, epoch):
    records, vfeats, vfeat_lens, w_lens, word_ids, char_ids, s_labels, e_labels, label = data 
    curr_batch_size = vfeats.size(0)
    start_entropies = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    start_values = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    start_log_probs = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    start_rewards = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    end_entropies = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    end_values = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    end_log_probs = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    end_rewards = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    Previous_IoUs = torch.zeros(configs.num_steps, curr_batch_size)
    Predict_IoUs = torch.zeros(configs.num_steps, curr_batch_size)
    Previous_dis = torch.zeros(configs.num_steps, curr_batch_size, 2).to(device)
    Predict_dis = torch.zeros(configs.num_steps, curr_batch_size, 2).to(device)
    locations = torch.zeros(configs.num_steps, curr_batch_size, 2)
    act_locations = torch.zeros(configs.num_steps, curr_batch_size, 2)
    evidences = torch.zeros(configs.num_steps, curr_batch_size, configs.n_rl)
    start_masks = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    start_mask = torch.ones(curr_batch_size).float().to(device)
    end_masks = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    end_mask = torch.ones(curr_batch_size).float().to(device)
    start_h_t = torch.zeros(curr_batch_size, configs.n_hidden).float().to(device)
    end_h_t = torch.zeros(curr_batch_size, configs.n_hidden).float().to(device)

    loc = torch.zeros(curr_batch_size, 2).float()
    gt_loc = (torch.stack((s_labels / vfeat_lens, e_labels / vfeat_lens),dim=1))
    loss_evi = 0
    edl_loss = EvidenceLoss(
            num_classes=configs.n_rl,
            evidence='exp',
            loss_type='log',
            with_kldiv=False,
            with_avuloss=False,
            disentangle=False,
            annealing_method='exp'
    )
    
    for step in range(configs.num_steps):
        if step == 0:
            if is_pos:
                loc[:, 0] = pos_start_point * torch.ones_like(loc[:, 0])
                loc[:, 1] = pos_end_point * torch.ones_like(loc[:, 1])
            else:
                loc[:, 0] = neg_start_point * torch.ones_like(loc[:, 0])
                loc[:, 1] = neg_end_point * torch.ones_like(loc[:, 1])
            visual_fea = None
            sen_fea = None 
        else:
            for i in range(curr_batch_size):
                if start_action[i] == 6:
                    start_mask[i] = 0
                if end_action[i] == 6:
                    end_mask[i] = 0

        # prepare features
        # query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device) video_mask = convert_length_to_mask(vfeat_lens).to(device)
        vfeats,word_ids  = vfeats.to(device),word_ids.to(device)
        pre_location = loc.clone()
        P_start = loc[:, 0].clone()
        P_end = loc[:, 1].clone()
        # generate mask
        #query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)
        #seg_mask = convert_length_to_mask(seq_lens).to(device)
        # compute logits
        pre_location = loc.clone().to(device) 
        sent = ini_model(word_ids)
        visual_fea, sen_fea, start_h_t, start_logit, start_v_t, end_h_t, \
                       end_logit, end_v_t, p_tIoU, p_loc, p_dis, relative_loc = model(step, vfeats, \
                              sent.detach(), visual_fea, sen_fea, pre_location.detach(), \
                              w_lens.detach(), vfeat_lens.detach(),\
                              start_h_t.detach(), end_h_t.detach())
        # entropy
        start_prob = F.softmax(start_logit, dim=1)
        start_log_prob = F.log_softmax(start_logit, dim=1)
        start_entropy = -(start_log_prob * start_prob).sum(1)
        start_entropies[step, :] = start_entropy
        
        start_action = start_prob.multinomial(num_samples=1).data
        start_log_prob = start_log_prob.gather(1, start_action)
        start_action = start_action.cpu().numpy()[:, 0]

        end_prob = F.softmax(end_logit, dim=1)
        end_log_prob = F.log_softmax(end_logit, dim=1)
        end_entropy = -(end_log_prob * end_prob).sum(1)
        end_entropies[step, :] = end_entropy

        end_action = end_prob.multinomial(num_samples=1).data
        end_log_prob = end_log_prob.gather(1, end_action)
        end_action = end_action.cpu().numpy()[:, 0]


        Predict_IoUs[step, :] = p_tIoU.squeeze(1)
        locations[step, :, :] = p_loc
        gl = gt_loc.expand(curr_batch_size, 2)
        Previous_dis[step, :, :] = gl - loc
        Predict_dis[step, :, :] = p_dis
            

        if step == 0:
            Previous_IoU = calculate_RL_IoU_batch(loc, gt_loc)
        else:
            Previous_IoU = current_IoU

        Previous_IoUs[step, :] = Previous_IoU
        loc = renew_state(loc, start_action, end_action, start_mask, end_mask, \
                            configs.delta0, configs.delta1, configs.delta2)
        act_locations[step, :, :] = loc
        current_IoU = calculate_RL_IoU_batch(loc, gt_loc)
        C_start = loc[:, 0].clone()
        C_end = loc[:, 1].clone()
        start_reward = calculate_reward_batch_withstop_start(P_start, C_start, C_end, gt_loc, \
                                        configs.beta, configs.threshold, start_action, configs.gamma)
        start_values[step, :] = start_v_t.squeeze(1)
        start_log_probs[step, :] = start_log_prob.squeeze(1)
        start_rewards[step, :] = start_reward
        start_masks[step, :] = start_mask

        end_reward = calculate_reward_batch_withstop_end(P_end, C_end, gt_loc, C_start, \
                                        configs.beta, configs.threshold, end_action, configs.gamma)
        end_values[step, :] = end_v_t.squeeze(1)
        end_log_probs[step, :] = end_log_prob.squeeze(1)
        end_rewards[step, :] = end_reward
        end_masks[step, :] = end_mask
        
        # Move
        gt_rloc_label = get_relative_loc(pre_location.cpu(), gt_loc, configs.delta0)
        edl_results = edl_loss(
            output=relative_loc,
            target=gt_rloc_label.to(device),
            epoch=epoch,
            total_epoch=configs.epochs
        )
        for b_idx in range(curr_batch_size):
            if start_mask[b_idx] == 1 or end_mask[b_idx] == 1:
                loss_evi += edl_results['loss_cls'][b_idx]
        
        evidences[step, :] = relative_loc
    #per_time = (time.time() - cur_time) / float(args.bs)
    # compute losses
    start_value_loss = 0
    end_policy_loss = 0
    end_value_loss = 0
    start_policy_loss = 0
    s_dis_loss = 0
    e_dis_loss = 0
    loc_loss = 0
    iou_loss = 0
    idx = 0

    mask_1 = torch.zeros_like(Previous_IoUs)
    
    
    for j in range(curr_batch_size):
        mask_start = start_masks[:, j]
        mask_end = end_masks[:, j]
        n_s = 0
        n_e = 0
        
        # mark the processed experiences
        for index in range(configs.num_steps):
            if mask_start[index] == 1:
                n_s = n_s + 1
            if mask_end[index] == 1:
                n_e = n_e + 1
    
        num = max(n_s, n_e)
        
        for k in reversed(range(num)):
                    
            sign_s = 0
            sign_e = 0

            if k == n_s - 1:
                S_R = configs.gamma * start_values[k][j] + start_rewards[k][j]
                sign_s = 1
            if k == n_e - 1:
                E_R = configs.gamma * end_values[k][j] + end_rewards[k][j]
                sign_e = 1
            if k < n_s - 1:
                S_R = configs.gamma * S_R + start_rewards[k][j]
                sign_s = 1
            if k < n_e - 1:
                E_R = configs.gamma * E_R + end_rewards[k][j]
                sign_e = 1
            
            if sign_s == 1:
                s_advantage = S_R.detach() - start_values[k][j]
                start_value_loss = start_value_loss + s_advantage.pow(2)
                start_policy_loss = start_policy_loss - start_log_probs[k][j] * s_advantage.detach() - configs.lambda_0 * start_entropies[k][j]
            if sign_e == 1:
                e_advantage = E_R.detach() - end_values[k][j]
                end_value_loss = end_value_loss + e_advantage.pow(2)
                end_policy_loss = end_policy_loss - end_log_probs[k][j] * e_advantage.detach() - configs.lambda_0 * end_entropies[k][j]

            iou_loss += torch.abs(Previous_IoUs[k, j] - Predict_IoUs[k, j])
            mask_1[k, j] = Previous_IoUs[k, j] > 0.4
            idx += 1
    
    start_policy_loss /= idx
    start_value_loss /= idx
    end_policy_loss /= idx
    end_value_loss /= idx
    iou_loss /= idx
    loss_evi /= idx

    loc_id = 0
    for i in range(len(mask_1)):
        for j in range(len(mask_1[i])):
            if mask_1[i, j] == 1:
                loc_loss += (torch.abs(gt_loc[j][0].detach() - locations[i][j][0]) +
                                torch.abs(gt_loc[j][1].detach() - locations[i][j][1])) / 2.0
            
                s_dis_loss += smoothloss(Predict_dis[i][j][0], Previous_dis[i][j][0].detach())
                e_dis_loss += smoothloss(Predict_dis[i][j][1], Previous_dis[i][j][1].detach())
                loc_id += 1

    dis_loss = s_dis_loss + e_dis_loss    
    if loc_id != 0:
        loc_loss /= loc_id
        dis_loss /= loc_id    
    
    policy_loss = start_policy_loss + end_policy_loss
    value_loss = start_value_loss + end_value_loss
    
    return policy_loss, value_loss, iou_loss, loc_loss, loss_evi, dis_loss, evidences, Predict_IoUs, loc, gt_loc, act_locations


def get_scan_loss(model, ini_model, data, epoch):
    records, vfeats, vfeat_lens, w_lens, word_ids, char_ids, s_labels, e_labels, label = data 
    curr_batch_size = vfeats.size(0)
    start_entropies = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    start_values = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    start_log_probs = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    start_rewards = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    end_entropies = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    end_values = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    end_log_probs = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    end_rewards = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    Previous_IoUs = torch.zeros(configs.num_steps, curr_batch_size)
    Predict_IoUs = torch.zeros(configs.num_steps, curr_batch_size)
    Previous_dis = torch.zeros(configs.num_steps, curr_batch_size, 2).to(device)
    Predict_dis = torch.zeros(configs.num_steps, curr_batch_size, 2).to(device)
    locations = torch.zeros(configs.num_steps, curr_batch_size, 2)
    act_locations = torch.zeros(configs.num_steps, curr_batch_size, 2)
    evidences = torch.zeros(configs.num_steps, curr_batch_size, configs.n_rl)
    start_masks = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    start_mask = torch.ones(curr_batch_size).float().to(device)
    end_masks = torch.zeros(configs.num_steps, curr_batch_size).to(device)
    end_mask = torch.ones(curr_batch_size).float().to(device)
    start_h_t = torch.zeros(curr_batch_size, configs.n_hidden).float().to(device)
    end_h_t = torch.zeros(curr_batch_size, configs.n_hidden).float().to(device)

    accept_loc = torch.zeros(curr_batch_size, 2).float()
    scan_loc = torch.zeros(curr_batch_size, 2).float()
    gt_loc = (torch.stack((s_labels / vfeat_lens, e_labels / vfeat_lens),dim=1))
    loss_evi = 0
    edl_loss = EvidenceLoss(
            num_classes=configs.n_rl,
            evidence='exp',
            loss_type='log',
            with_kldiv=False,
            with_avuloss=False,
            disentangle=False,
            annealing_method='exp'
    )
    
    for step in range(configs.num_steps):
        if step == 0:
            scan_loc[:, 0] = scan_start_point * torch.ones_like(scan_loc[:, 0])
            scan_loc[:, 1] = scan_end_point * torch.ones_like(scan_loc[:, 1])
            accept_loc = scan_loc.clone()
            visual_fea = None
            sen_fea = None  
        else:
            for i in range(curr_batch_size):
                if start_action[i] == STOP_ACTION_SCAN:
                    start_mask[i] = 0
                if end_action[i] == STOP_ACTION_SCAN:
                    end_mask[i] = 0

        # prepare features
        # query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device) video_mask = convert_length_to_mask(vfeat_lens).to(device)
        vfeats,word_ids  = vfeats.to(device),word_ids.to(device)
        pre_scan_location = scan_loc.clone()
        P_start = accept_loc[:, 0].clone()
        P_end = accept_loc[:, 1].clone()
        # generate mask
        #query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)
        #seg_mask = convert_length_to_mask(seq_lens).to(device)
        # compute logits
        pre_scan_location = pre_scan_location.clone().to(device) 
        sent = ini_model(word_ids)
        visual_fea, sen_fea, start_h_t, start_logit, start_v_t, end_h_t, \
                       end_logit, end_v_t, p_tIoU, p_loc, p_dis, relative_loc = model(step, vfeats, \
                              sent.detach(), visual_fea, sen_fea, pre_scan_location.detach(), \
                              w_lens.detach(), vfeat_lens.detach(),\
                              start_h_t.detach(), end_h_t.detach())
        # entropy
        start_prob = F.softmax(start_logit, dim=1)
        start_log_prob = F.log_softmax(start_logit, dim=1)
        start_entropy = -(start_log_prob * start_prob).sum(1)
        start_entropies[step, :] = start_entropy
        
        start_action = start_prob.multinomial(num_samples=1).data
        start_log_prob = start_log_prob.gather(1, start_action)
        start_action = start_action.cpu().numpy()[:, 0]

        end_prob = F.softmax(end_logit, dim=1)
        end_log_prob = F.log_softmax(end_logit, dim=1)
        end_entropy = -(end_log_prob * end_prob).sum(1)
        end_entropies[step, :] = end_entropy

        end_action = end_prob.multinomial(num_samples=1).data
        end_log_prob = end_log_prob.gather(1, end_action)
        end_action = end_action.cpu().numpy()[:, 0]


        Predict_IoUs[step, :] = p_tIoU.squeeze(1)
        locations[step, :, :] = p_loc
        gl = gt_loc.expand(curr_batch_size, 2)
        Previous_dis[step, :, :] = gl - scan_loc
        Predict_dis[step, :, :] = p_dis
            

        if step == 0:
            Previous_IoU = calculate_RL_IoU_batch(scan_loc, gt_loc)
        else:
            Previous_IoU = current_IoU

        Previous_IoUs[step, :] = Previous_IoU
        scan_loc, accept_loc = renew_state_scan(scan_loc, accept_loc, start_action, end_action, start_mask, end_mask, \
                            configs.sd0, configs.sd1, configs.sd2, configs.move)
        act_locations[step, :, :] = scan_loc
        current_IoU = calculate_RL_IoU_batch(scan_loc, gt_loc)
        C_start = accept_loc[:, 0].clone()
        C_end = accept_loc[:, 1].clone()
        start_reward = calculate_reward_batch_withstop_start_scan(P_start, C_start, C_end, gt_loc, start_action, \
                                        configs.gamma, configs.beta, configs.scan_reward_value)
        start_values[step, :] = start_v_t.squeeze(1)
        start_log_probs[step, :] = start_log_prob.squeeze(1)
        start_rewards[step, :] = start_reward
        start_masks[step, :] = start_mask

        end_reward = calculate_reward_batch_withstop_end_scan(P_end, C_end, C_start, gt_loc, end_action, \
                                        configs.gamma, configs.beta, configs.scan_reward_value)
        end_values[step, :] = end_v_t.squeeze(1)
        end_log_probs[step, :] = end_log_prob.squeeze(1)
        end_rewards[step, :] = end_reward
        end_masks[step, :] = end_mask
        
        # Move
        gt_rloc_label = get_relative_loc(pre_scan_location.cpu(), gt_loc, configs.sd0)
        edl_results = edl_loss(
            output=relative_loc,
            target=gt_rloc_label.to(device),
            epoch=epoch,
            total_epoch=configs.epochs
        )
        for b_idx in range(curr_batch_size):
            if start_mask[b_idx] == 1 or end_mask[b_idx] == 1:
                loss_evi += edl_results['loss_cls'][b_idx]
        
        evidences[step, :] = relative_loc
    #per_time = (time.time() - cur_time) / float(args.bs)
    # compute losses
    start_value_loss = 0
    end_policy_loss = 0
    end_value_loss = 0
    start_policy_loss = 0
    s_dis_loss = 0
    e_dis_loss = 0
    loc_loss = 0
    iou_loss = 0
    idx = 0

    mask_1 = torch.zeros_like(Previous_IoUs)
    
    
    for j in range(curr_batch_size):
        mask_start = start_masks[:, j]
        mask_end = end_masks[:, j]
        n_s = 0
        n_e = 0
        
        # mark the processed experiences
        for index in range(configs.num_steps):
            if mask_start[index] == 1:
                n_s = n_s + 1
            if mask_end[index] == 1:
                n_e = n_e + 1
    
        num = max(n_s, n_e)
        
        for k in reversed(range(num)):
                    
            sign_s = 0
            sign_e = 0

            if k == n_s - 1:
                S_R = configs.gamma * start_values[k][j] + start_rewards[k][j]
                sign_s = 1
            if k == n_e - 1:
                E_R = configs.gamma * end_values[k][j] + end_rewards[k][j]
                sign_e = 1
            if k < n_s - 1:
                S_R = configs.gamma * S_R + start_rewards[k][j]
                sign_s = 1
            if k < n_e - 1:
                E_R = configs.gamma * E_R + end_rewards[k][j]
                sign_e = 1
            
            if sign_s == 1:
                s_advantage = S_R.detach() - start_values[k][j]
                start_value_loss = start_value_loss + s_advantage.pow(2)
                start_policy_loss = start_policy_loss - start_log_probs[k][j] * s_advantage.detach() - configs.lambda_0 * start_entropies[k][j]
            if sign_e == 1:
                e_advantage = E_R.detach() - end_values[k][j]
                end_value_loss = end_value_loss + e_advantage.pow(2)
                end_policy_loss = end_policy_loss - end_log_probs[k][j] * e_advantage.detach() - configs.lambda_0 * end_entropies[k][j]

            iou_loss += torch.abs(Previous_IoUs[k, j] - Predict_IoUs[k, j])
            mask_1[k, j] = Previous_IoUs[k, j] > 0.4
            idx += 1
    
    start_policy_loss /= idx
    start_value_loss /= idx
    end_policy_loss /= idx
    end_value_loss /= idx
    iou_loss /= idx
    loss_evi /= idx

    loc_id = 0
    for i in range(len(mask_1)):
        for j in range(len(mask_1[i])):
            if mask_1[i, j] == 1:
                loc_loss += (torch.abs(gt_loc[j][0].detach() - locations[i][j][0]) +
                                torch.abs(gt_loc[j][1].detach() - locations[i][j][1])) / 2.0
            
                s_dis_loss += smoothloss(Predict_dis[i][j][0], Previous_dis[i][j][0].detach())
                e_dis_loss += smoothloss(Predict_dis[i][j][1], Previous_dis[i][j][1].detach())
                loc_id += 1

    dis_loss = s_dis_loss + e_dis_loss    
    if loc_id != 0:
        loc_loss /= loc_id
        dis_loss /= loc_id    
    
    policy_loss = start_policy_loss + end_policy_loss
    value_loss = start_value_loss + end_value_loss
    
    return policy_loss, value_loss, iou_loss, loc_loss, loss_evi, dis_loss, evidences, Predict_IoUs, scan_loc, accept_loc, gt_loc, act_locations

# train and test



def update_loss_data(data, policy_loss, value_loss, iou_loss, loc_loss, evi_loss, dis_loss):
    data['policy_loss_sum'] += float(policy_loss)
    data['value_loss_sum'] += float(value_loss)
    data['iou_loss_sum'] += float(iou_loss)
    data['loc_loss_sum'] += float(loc_loss)
    data['evi_loss_sum'] += float(evi_loss)
    data['dis_loss_sum'] += float(dis_loss)
    #data['total_rewards_epoch'] += float(reward_sum)
    return data

def print_loss_data(data, epoch, iteration):
    print("Train Epoch: %d | Index: %d | policy loss: %f" % (epoch, iteration+1, data['policy_loss_sum']))            
    print("Train Epoch: %d | Index: %d | value loss: %f" % (epoch, iteration+1, data['value_loss_sum'])) 
    print("Train Epoch: %d | Index: %d | IoU loss: %f" % (epoch, iteration+1, data['iou_loss_sum'] )) 
    print("Train Epoch: %d | Index: %d | Location loss: %f" % (epoch, iteration+1, data['loc_loss_sum']))
    print("Train Epoch: %d | Index: %d | Distance loss: %f" % (epoch, iteration+1, data['dis_loss_sum']))
    print("Train Epoch: %d | Index: %d | Evidence loss: %f" % (epoch, iteration+1, data['evi_loss_sum']))
    #print("Train Epoch: %d | Index: %d | Reward: %f" % (epoch, iteration+1, data['total_rewards_epoch']))



if configs.mode.lower() == 'train':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    eval_period = num_train_batches // 2
    save_json(vars(configs), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)
    # build model
    neg_model = NegModels(configs).to(device)
    pos_model = PosModels(configs).to(device)
    scan_model = ScanModels(configs).to(device)
    man_model = ManagerModel2(configs).to(device)
    ini_model = WordEmbedding(configs.word_size, configs.word_dim, configs.drop_rate, dataset['word_vector']).to(device)
    ini_model.eval()
    man_loss = nn.MSELoss()
    smoothloss = torch.nn.SmoothL1Loss().to(device)
    optimizer = torch.optim.Adam(list(scan_model.parameters()) + list(neg_model.parameters()) + list(pos_model.parameters()) + list(man_model.parameters()), lr=configs.init_lr) 
    # start training
    best_r1i7 = -1.0
    all_run = 1
    print('start training...', flush=True)
    iteration = 0
    max_select_iou5 = 0.0
    max_pos_iou5 = 0.0
    max_neg_iou5 = 0.0
    max_scan_iou5 = 0.0
    for epoch in range(configs.epochs):
        loss_sum = 0
        select_loss_sum = 0
        pos_loss_data = {"policy_loss_sum":0, \
            "value_loss_sum":0, "iou_loss_sum":0, "loc_loss_sum":0, "evi_loss_sum":0, "dis_loss_sum":0,  "total_rewards_epoch":0}
        neg_loss_data = pos_loss_data.copy()
        scan_loss_data = pos_loss_data.copy()
        count = 0
        for data in train_loader:
            neg_model.train()
            pos_model.train()
            scan_model.train()
            man_model.train()
            #count += 1
            #if count > 2:
            #    break
            neg_policy_loss, neg_value_loss, neg_iou_loss, neg_loc_loss, neg_evi_loss, neg_dis_loss, neg_evi, neg_iou, neg_loc, gt_loc, neg_loc_save = \
                get_loss(neg_model, ini_model, data, False, epoch)
            pos_policy_loss, pos_value_loss, pos_iou_loss, pos_loc_loss, pos_evi_loss, pos_dis_loss, pos_evi, pos_iou, pos_loc, gt_loc, pos_loc_save = \
                get_loss(pos_model, ini_model, data, True, epoch)
            scan_policy_loss, scan_value_loss, scan_iou_loss, scan_loc_loss, scan_evi_loss, scan_dis_loss, scan_evi, scan_ious, scan_loc, accept_loc, gt_loc, scan_loc_save = \
                get_scan_loss(scan_model, ini_model, data, epoch)
            pos_data = (pos_iou.detach().to(device), pos_evi.detach().to(device), pos_loc_save.detach().to(device))
            neg_data = (neg_iou.detach().to(device), neg_evi.detach().to(device), neg_loc_save.detach().to(device))
            scan_data = (scan_ious.detach().to(device), scan_evi.detach().to(device), scan_loc_save.detach().to(device))
            pos_logit = man_model(pos_data, pos_loc.detach().to(device))
            neg_logit = man_model(neg_data, neg_loc.detach().to(device))
            scan_logit = man_model(scan_data, accept_loc.detach().to(device))
 
            pos_tiou, neg_tiou, scan_tiou = get_ious(gt_loc, pos_loc, neg_loc, accept_loc)
            pos_tiou, neg_tiou, scan_tiou = pos_tiou.to(device), neg_tiou.to(device), scan_tiou.to(device)
            neg_all_loss = neg_policy_loss + configs.lambda_1 * neg_value_loss + configs.lambda_3 *(neg_iou_loss + configs.lambda_2 * neg_loc_loss) + neg_dis_loss + neg_evi_loss
            pos_all_loss = pos_policy_loss + configs.lambda_1 * pos_value_loss + configs.lambda_3 *(pos_iou_loss + configs.lambda_2 * pos_loc_loss) + pos_dis_loss + pos_evi_loss
            scan_all_loss = scan_policy_loss + configs.lambda_1 * scan_value_loss + configs.lambda_3 *(scan_iou_loss + configs.lambda_2 * scan_loc_loss) + scan_dis_loss + scan_evi_loss
            scan_loss_data = \
                update_loss_data(scan_loss_data, \
                    scan_policy_loss, scan_value_loss, scan_iou_loss, scan_loc_loss, scan_evi_loss, scan_loc_loss)
            pos_loss_data = \
                update_loss_data(pos_loss_data, \
                    pos_policy_loss, pos_value_loss, pos_iou_loss, pos_loc_loss, pos_evi_loss, pos_loc_loss)
            neg_loss_data = \
                update_loss_data(neg_loss_data, \
                    neg_policy_loss, neg_value_loss, neg_iou_loss, neg_loc_loss, neg_evi_loss, neg_dis_loss)
            optimizer.zero_grad()
            select_loss = man_loss(pos_logit, pos_tiou) + man_loss(neg_logit, neg_tiou) + man_loss(scan_logit, scan_tiou)
            all_loss = neg_all_loss + pos_all_loss + select_loss + scan_all_loss            
            #all_loss =  pos_all_loss
            all_loss.backward(retain_graph = True)
            loss_sum += all_loss.cpu().data
            adjust_lr(optimizer, configs.init_lr, configs.warmup_init_lr, configs.warmup_updates, all_run)
            optimizer.step()
            iteration = iteration + 1
            all_run += 1
            if all_run % configs.n_save == 0:
                if val_loader != None:
                    m_selected_model_new(pos_model, neg_model, scan_model, man_model, ini_model, val_loader)
                else:
                    m_selected_model_new(pos_model, neg_model, scan_model, man_model, ini_model, test_loader)
                torch.save(pos_model.state_dict(), os.path.join(model_dir, 'pos_{}_{}.t7'.format(configs.model_name, all_run)))
                torch.save(neg_model.state_dict(), os.path.join(model_dir, 'neg_{}_{}.t7'.format(configs.model_name, all_run)))
                torch.save(scan_model.state_dict(), os.path.join(model_dir, 'scan_{}_{}.t7'.format(configs.model_name, all_run)))
                torch.save(man_model.state_dict(), os.path.join(model_dir, 'man_{}_{}.t7'.format(configs.model_name, all_run)))
                print('save the %d model' % iteration)
            select_loss_sum += float(select_loss.cpu().data)
        print(loss_sum)
        print("Select loss: " + str(float(select_loss_sum)))
        print("Positive loss")
        print_loss_data(pos_loss_data, epoch, iteration)
        print("Negative loss")
        print_loss_data(neg_loss_data, epoch, iteration)
        print("Scan loss")
        print_loss_data(scan_loss_data, epoch, iteration)
        
