" Some useful functions "

import numpy as np
import operator
import torch
import torch.nn as nn


def get_relative_loc(real_loc, curr_loc, loc_threshold, n_evi = 16):
    loc_label_batch = torch.zeros(len(real_loc),n_evi)
    for i in range(len(real_loc)):
        r_s,r_e = real_loc[i].tolist()
        c_s,c_e = curr_loc[i].tolist()
        s_label = relative_loc_label(r_s, c_s, loc_threshold)
        e_label = relative_loc_label(r_e, c_e, loc_threshold)
        curr_mat = torch.zeros(4,4).float()
        curr_mat[s_label][e_label] = 1.0
        loc_label_batch[i, :] = curr_mat.view(-1)
    return loc_label_batch
def relative_loc_label(r, c, loc_threshold):
    if r - c > loc_threshold:
        return 0
    elif c - r  > loc_threshold:
        return 1
    elif r - c <= loc_threshold and r - c >= 0:
        return 2
    else:
        return 3

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        #nn.init.orthogonal_(m.weight.data, 1.0)
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
def renew_state(pre_location, a_s, a_e, s_mask, e_mask, delta0, delta1, delta2):
    bs = len(a_s)
    location = pre_location.clone()
    for i in range(bs):
        start = float(pre_location[i, 0])
        end = float(pre_location[i, 1])
        if s_mask[i] == 1:
            if a_s[i] == 0:
                start = start + delta0
            elif a_s[i] == 1:
                start = start + delta1
            elif a_s[i] == 2:
                start = start + delta2
            elif a_s[i] == 3:
                start = start - delta0
            elif a_s[i] == 4:
                start = start - delta1
            elif a_s[i] == 5:
                start = start - delta2

        if e_mask[i] == 1:
            if a_e[i] == 0:
                end = end + delta0
            elif a_e[i] == 1:
                end = end + delta1
            elif a_e[i] == 2:
                end = end + delta2
            elif a_e[i] == 3:
                end = end - delta0
            elif a_e[i] == 4:
                end = end - delta1
            elif a_e[i] == 5:
                end = end - delta2
           
        if start < 0:
            start = 0
        if end > 1:
            end = 1
           
        location[i, 0] = start
        location[i, 1] = end
          
    return location

def renew_state_test(pre_location, a_s, a_e, delta0, delta1, delta2, m_s, m_e):
    location = pre_location.clone()
    start = pre_location[0, 0]
    end = pre_location[0, 1]
    if m_s == 1:
        if a_s == 0:
            start = start + delta0
        elif a_s == 1:
            start = start + delta1
        elif a_s == 2:
            start = start + delta2
        elif a_s == 3:
            start = start - delta0
        elif a_s == 4:
            start = start - delta1
        elif a_s == 5:
            start = start - delta2
    if m_e == 1:
        if a_e == 0:
            end = end + delta0
        elif a_e == 1:
            end = end + delta1
        elif a_e == 2:
            end = end + delta2
        elif a_e == 3:
            end = end - delta0
        elif a_e == 4:
            end = end - delta1
        elif a_e == 5:
            end = end - delta2
   
    if start < 0:
        start = 0
    if end > 1:
        end = 1
    
    location[0, 0] = start
    location[0, 1] = end
 
    return location


def calculate_reward_batch_withstop_start(Previous_start, Current_start, Current_end, gt_loc, beta, threshold, s_a, gamma):
    batch_size = len(Previous_start)
    reward = torch.zeros(batch_size)

    for i in range(batch_size):
        gt_start = float(gt_loc[i, 0])
        P_dis = abs(Previous_start[i] - gt_start)
        C_dis = abs(Current_start[i] - gt_start)
        if s_a[i] != 6:
            if Current_start[i] >= 0 and Current_start[i] < Current_end[i]:
                if P_dis > C_dis:
                    reward[i] = 1 - C_dis  
                else:
                    reward[i] = 0
            else:
                reward[i] = beta
            reward[i] = float(reward[i]) + P_dis - gamma * C_dis
        else:
            if C_dis <= threshold:
                reward[i] = 1
            else:
                reward[i] = -1
    return reward

def calculate_reward_batch_withstop_end(Previous_end, Current_end, gt_loc, Current_start, beta, threshold, e_a, gamma):
    batch_size = len(Previous_end)
    reward = torch.zeros(batch_size)

    for i in range(batch_size):
        gt_end = float(gt_loc[i, 1])
        P_dis = abs(Previous_end[i] - gt_end)
        C_dis = abs(Current_end[i] - gt_end)
        if e_a[i] != 6:
            if Current_end[i] > Current_start[i] and Current_end[i] <= 1:
                if P_dis > C_dis:
                    reward[i] = 1 - C_dis  
                else:
                    reward[i] = 0
            else:
                reward[i] = beta
            reward[i] = float(reward[i]) + P_dis - gamma * C_dis
        else:
            if C_dis <= threshold:
                reward[i] = 1
            else:
                reward[i] = -1
    return reward

def get_relative_loc(real_loc, curr_loc):
    tiou = calculate_RL_IoU_batch(real_loc, curr_loc)
    loc_label_batch = torch.zeros(len(tiou),3)
    for i in range(len(tiou)):
        if tiou[i] <= 0:
            if real_loc[i][1] <= curr_loc[i][0]:
                loc_label_batch[i,0] = 1.0
            else:
                loc_label_batch[i,1] = 1.0
        else:
            loc_label_batch[i,2] = 1.0
    return loc_label_batch
    
def get_best_decision(real_loc, pos_loc, neg_loc):
    pos_tiou = calculate_RL_IoU_batch(real_loc, pos_loc)
    neg_tiou = calculate_RL_IoU_batch(real_loc, neg_loc)
    loc_label_batch = torch.zeros(len(pos_tiou),dtype=torch.int64)
    for i in range(len(pos_tiou)):
        if pos_tiou[i] >= neg_tiou[i]:
            loc_label_batch[i] = 1
        else:
            loc_label_batch[i] = 0
    return loc_label_batch


def calculate_reward_batch_manager(m_assignment, gt_assignment):
    batch_size = len(m_assignment[0])
    reward = torch.zeros(batch_size)
    for i in range(batch_size):
        curr_seq = m_assignment[:,i]
        curr_rewards = []
        for j in curr_seq:
            if gt_assignment[j, i] == m_assignment[j, i]:
                if gt_assignment[j, i] == 0:
                    curr_rewards.append(1)
                elif gt_assignment[j, i] == 1:
                    curr_rewards.append(1)
                elif gt_assignment[j, i] == 2:
                    curr_rewards.append(0.5)
            else:
                curr_rewards.append(-1)
        reward[i] = np.mean(curr_rewards)
    return reward
def calculate_reward_batch_withstop(Previou_IoU, current_IoU, t):
    batch_size = len(Previou_IoU)
    reward = torch.zeros(batch_size)

    for i in range(batch_size):
        if current_IoU[i] > Previou_IoU[i] and Previou_IoU[i]>=0:
            reward[i] = 1 -0.001*t
        elif current_IoU[i] <= Previou_IoU[i] and current_IoU[i]>=0:
            reward[i] = -0.001*t
        else:
            reward[i] = -1 -0.001*t
    return reward


def calculate_reward(Previou_IoU, current_IoU, t):
    if current_IoU > Previou_IoU and Previou_IoU>=0:
        reward = 1-0.001*t
    elif current_IoU <= Previou_IoU and current_IoU>=0:
        reward = -0.001*t
    else:
        reward = -1-0.001*t

    return reward

def calculate_RL_IoU_batch(i0, i1):
    # calculate temporal intersection over union
    batch_size = len(i0)
    iou_batch = torch.zeros(batch_size)

    for i in range(len(i0)):
        union = (min(i0[i][0], i1[i][0]), max(i0[i][1], i1[i][1]))
        inter = (max(i0[i][0], i1[i][0]), min(i0[i][1], i1[i][1]))
        # if inter[1] < inter[0]:
        #     iou = 0
        # else:
        iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
        iou_batch[i] = iou
    return iou_batch

def calculate_IoU(i0, i1):
    # calculate temporal intersection over union
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou


def nms_temporal(x1,x2,s, overlap):
    pick = []
    assert len(x1)==len(s)
    assert len(x2)==len(s)
    if len(x1)==0:
        return pick

    union = map(operator.sub, x2, x1) # union = x2-x1

    I = [i[0] for i in sorted(enumerate(s), key=lambda x:x[1])] # sort and get index

    while len(I)>0:
        i = I[-1]
        pick.append(i)

        xx1 = [max(x1[i],x1[j]) for j in I[:-1]]
        xx2 = [min(x2[i],x2[j]) for j in I[:-1]]
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u]/(union[i] + union[I[u]] - inter[u]) for u in range(len(I)-1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <=overlap:
                I_new.append(I[j])
        I = I_new
    return pick


def compute_IoU_recall_top_n_forreg_rl(top_n, iou_thresh, sentence_image_reg_mat, sclips):
    correct_num = 0.0
    for k in range(sentence_image_reg_mat.shape[0]):
        gt = sclips[k]
        # print(gt)
        gt_start = float(gt.split("_")[1])
        gt_end = float(gt.split("_")[2])

        pred_start = sentence_image_reg_mat[k, 0]
        pred_end = sentence_image_reg_mat[k, 1]
        iou = calculate_IoU((gt_start, gt_end),(pred_start, pred_end))
        if iou>=iou_thresh:
            correct_num+=1

    return correct_num

def compute_IoU_recall_top_n_forreg(top_n, iou_thresh, sentence_image_mat, sentence_image_reg_mat, sclips, iclips):
    correct_num = 0.0
    for k in range(sentence_image_mat.shape[0]):
        gt = sclips[k]
        gt_start = float(gt.split("_")[1])
        gt_end = float(gt.split("_")[2])
        #print gt +" "+str(gt_start)+" "+str(gt_end)
        sim_v = [v for v in sentence_image_mat[k]]
        starts = [s for s in sentence_image_reg_mat[k,:,0]]
        ends = [e for e in sentence_image_reg_mat[k,:,1]]
        picks = nms_temporal(starts,ends, sim_v, iou_thresh-0.05)
        #sim_argsort=np.argsort(sim_v)[::-1][0:top_n]
        if top_n<len(picks): picks=picks[0:top_n]
        for index in picks:
            pred_start = sentence_image_reg_mat[k, index, 0]
            pred_end = sentence_image_reg_mat[k, index, 1]
            iou = calculate_IoU((gt_start, gt_end),(pred_start, pred_end))
            if iou>=iou_thresh:
                correct_num+=1
                break
    return correct_num
def adjust_lr(optimizer, lr, warmup_init_lr, warmup_updates, num_updates):
    warmup_end_lr = lr
    if warmup_init_lr < 0:
        warmup_init_lr = warmup_end_lr
    lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates
    decay_factor = warmup_end_lr * warmup_updates**0.5
    if num_updates < warmup_updates:
        lr = warmup_init_lr + num_updates*lr_step
    else:
        lr = decay_factor * num_updates**-0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def renew_state_scan(pre_scan_location, pre_accept_location, a_s, a_e, s_mask, e_mask, delta0, delta1, delta2, move_step):
    bs = len(a_s)
    scan_location = pre_scan_location.clone()
    accept_location = pre_accept_location.clone()
    for i in range(bs):
        scan_start = float(scan_location[i, 0])
        scan_end = float(scan_location[i, 1])
        
        scan_mid = float((scan_end + scan_start) / 2)

        accept_start = float(accept_location[i, 0])
        accept_end = float(accept_location[i, 1])
        if s_mask[i] == 1:
            if a_s[i] == 0:
                accept_start = scan_mid + delta0 / 2
            elif a_s[i] == 1:
                accept_start = scan_mid + delta1 / 2
            elif a_s[i] == 2:
                accept_start = scan_mid + delta2 / 2
            elif a_s[i] == 3:
                accept_start = scan_mid - delta0 / 2
            elif a_s[i] == 4:
                accept_start = scan_mid - delta1 / 2
            elif a_s[i] == 5:
                accept_start = scan_mid - delta2 / 2

        if e_mask[i] == 1:
            if a_e[i] == 0:
                accept_end = scan_mid + delta0 / 2
            elif a_e[i] == 1:
                accept_end = scan_mid + delta1 / 2
            elif a_e[i] == 2:
                accept_end = scan_mid + delta2 / 2
            elif a_e[i] == 3:
                accept_end = scan_mid - delta0 / 2
            elif a_e[i] == 4:
                accept_end = scan_mid - delta1 / 2
            elif a_e[i] == 5:
                accept_end = scan_mid - delta2 / 2
           
        if accept_start < 0:
            accept_start = 0
        if accept_end > 1:
            accept_end = 1
           
        accept_location[i, 0] = accept_start
        accept_location[i, 1] = accept_end
           
        if s_mask[i] == 1:
            scan_start = scan_start + move_step
            scan_end = scan_end + move_step
        if scan_start < 0:
            scan_start = 0
        if scan_end > 1:
            scan_end = 1
           
        scan_location[i, 0] = scan_start
        scan_location[i, 1] = scan_end
          
    return scan_location, accept_location 
def renew_state_scan_test(pre_scan_location, pre_accept_location, a_s, a_e, delta0, delta1, delta2, move_step, m_s, m_e):
    scan_location = pre_scan_location.clone()
    accept_location = pre_accept_location.clone()
    scan_start = scan_location[0, 0]
    scan_end = scan_location[0, 1]
    accept_start = accept_location[0, 0]
    accept_end = accept_location[0, 1]
    scan_mid = float((scan_end + scan_start) / 2)
    if m_s == 1:
        if a_s == 0:
            accept_start = scan_mid + delta0 / 2
        elif a_s == 1:
            accept_start = scan_mid + delta1 / 2
        elif a_s == 2:
            accept_start = scan_mid + delta2 / 2
        elif a_s == 3:
            accept_start = scan_mid - delta0 / 2
        elif a_s == 4:
            accept_start = scan_mid - delta1 / 2
        elif a_s == 5:
            accept_start = scan_mid - delta2 / 2
    if m_e == 1:
        if a_e == 0:
            accept_end = scan_mid + delta0 / 2
        elif a_e == 1:
            accept_end = scan_mid + delta1 / 2
        elif a_e == 2:
            accept_end = scan_mid + delta2 / 2
        elif a_e == 3:
            accept_end = scan_mid - delta0 / 2
        elif a_e == 4:
            accept_end = scan_mid - delta1 / 2
        elif a_e == 5:
            accept_end = scan_mid - delta2 / 2

    if accept_start < 0:
        accept_start = 0
    if accept_end > 1:
        accept_end = 1
    accept_location[0, 0] = accept_start
    accept_location[0, 1] = accept_end

    if m_s == 1:
        scan_start = scan_start + move_step
        scan_end = scan_end + move_step
    if scan_start < 0:
        scan_start = 0
    if scan_end > 1:
        scan_end = 1
    scan_location[0, 0] = scan_start
    scan_location[0, 1] = scan_end
 
    return scan_location, accept_location


def calculate_reward_batch_withstop_start_scan(Previous_accept_start, Current_accept_start, Current_accept_end, gt_loc, s_a, gamma, beta, scan_reward, hold_action = 6):
    batch_size = len(Previous_accept_start)
    reward = torch.zeros(batch_size)

    for i in range(batch_size):
        gt_start = float(gt_loc[i, 0])
        P_dis = abs(Previous_accept_start[i] - gt_start)
        C_dis = abs(Current_accept_start[i] - gt_start)
        if s_a[i] == hold_action:
            reward[i] = scan_reward
        else:
            if Current_accept_start[i] >= 0 and Current_accept_start[i] < Current_accept_end[i]:
                if P_dis > C_dis:
                    reward[i] = 1 - C_dis  
                else:
                    reward[i] = -1.0 - C_dis
            else:
                reward[i] = beta
            reward[i] = float(reward[i]) + P_dis - gamma * C_dis
        
    return reward
def calculate_reward_batch_withstop_end_scan(Previous_accept_end, Current_accept_end, Current_accept_start, gt_loc, e_a, gamma, beta, scan_reward, hold_action = 6):
    batch_size = len(Previous_accept_end)
    reward = torch.zeros(batch_size)

    for i in range(batch_size):
        gt_end = float(gt_loc[i, 1])
        P_dis = abs(Previous_accept_end[i] - gt_end)
        C_dis = abs(Current_accept_end[i] - gt_end)
        if e_a[i] == hold_action:
            reward[i] = scan_reward
        else:
            if Current_accept_end[i] > Current_accept_start[i] and Current_accept_end[i] <= 1:
                if P_dis > C_dis:
                    reward[i] = 1 - C_dis  
                else:
                    reward[i] = -1.0 - C_dis
            else:
                reward[i] = beta
            reward[i] = float(reward[i]) + P_dis - gamma * C_dis
        
    return reward
def get_all_trust(real_loc, pos_loc, neg_loc, scan_loc, p_pos, p_neg, p_scan):
    pos_tiou, neg_tiou, scan_tiou = get_ious(real_loc, pos_loc, neg_loc, scan_loc)
    pos_t, neg_t, scan_t =  get_trust(pos_tiou, p_pos), get_trust(neg_tiou, p_neg), get_trust(scan_tiou, p_scan)
    return pos_t, neg_t, scan_t

def get_ious(real_loc, pos_loc, neg_loc, scan_loc):
    pos_tiou = calculate_RL_IoU_batch(real_loc, pos_loc)
    neg_tiou = calculate_RL_IoU_batch(real_loc, neg_loc)
    scan_tiou = calculate_RL_IoU_batch(real_loc, scan_loc)
    return pos_tiou, neg_tiou, scan_tiou
def get_trust(pre_iou, true_iou):
    diff = torch.abs(pre_iou, true_iou)
    return diff

def calculate_RL_IoU_batch(i0, i1):
    # calculate temporal intersection over union
    batch_size = len(i0)
    iou_batch = torch.zeros(batch_size)

    for i in range(len(i0)):
        union = (min(i0[i][0], i1[i][0]), max(i0[i][1], i1[i][1]))
        inter = (max(i0[i][0], i1[i][0]), min(i0[i][1], i1[i][1]))
        # if inter[1] < inter[0]:
        #     iou = 0
        # else:
        iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
        iou_batch[i] = iou
    return iou_batch