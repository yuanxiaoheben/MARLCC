import torch
import torch.nn as nn
import torch.nn.functional as F


class ManagerModel(nn.Module):
    '''
    Produce query interaction, decide 
    '''
    def __init__(self, configs):
        super(ManagerModel, self).__init__()
        self.d_rnn = nn.GRU(128, configs.n_inter, 2, \
                            batch_first=False, bidirectional=True, dropout=configs.drop_rate)
        self.out_linear = nn.Linear(configs.n_inter * 2 , 1)
        self.loc_ffn = nn.Linear(2, 64)
        self.iou_ffn = nn.Linear(1, 32)
        self.evi_ffn = nn.Linear(configs.n_rl, 32)
        self.drop_out = nn.Dropout(p=configs.drop_rate)
    def forward(self, data_1):
        f1 = self.seq_decision(data_1)
        out = self.choice(f1)
        return out.squeeze(-1)
    
    def seq_decision(self, data):
        iou_seq, evi_seq, loc_seq = data
        iou_seq = iou_seq.unsqueeze(-1)
        out,_ = self.d_rnn(torch.cat((self.iou_ffn(iou_seq), self.evi_ffn(evi_seq), self.loc_ffn(loc_seq)), dim=2))
        return out
    
    def choice(self, f1):
        out = self.out_linear(F.relu(self.drop_out(f1[-1])))
        return out

class ManagerModel2(nn.Module):
    '''
    Produce query interaction, decide 
    '''
    def __init__(self, configs):
        super(ManagerModel2, self).__init__()
        self.d_rnn = nn.GRU(128, configs.n_inter, 2, \
                            batch_first=False, bidirectional=True, dropout=configs.drop_rate)
        self.out_linear = nn.Linear(configs.n_inter * 2 , 1)
        self.loc_ffn = nn.Linear(2, 64)
        self.iou_ffn = nn.Linear(1, 32)
        self.evi_ffn = nn.Linear(configs.n_rl, 32)
        self.fc = nn.Linear(configs.n_inter * 2 + 64 , configs.n_inter * 2)
        self.drop_out = nn.Dropout(p=configs.drop_rate)
    def forward(self, data_1, final_loc):
        f1 = self.seq_decision(data_1)
        out = self.choice(f1, final_loc)
        return out.squeeze(-1)
    
    def seq_decision(self, data):
        iou_seq, evi_seq, loc_seq = data
        iou_seq = iou_seq.unsqueeze(-1)
        out,_ = self.d_rnn(torch.cat((self.iou_ffn(iou_seq), self.evi_ffn(evi_seq), self.loc_ffn(loc_seq)), dim=2))
        return out
    
    def choice(self, f1, final_loc):
        out = self.fc(torch.cat((self.loc_ffn(final_loc), f1[-1]), dim=-1))
        out = self.out_linear(F.relu(self.drop_out(out)))
        return out