import numpy as np
import torch
import torch.utils.data
from util.data_util import pad_seq, pad_video_seq, pad_char_seq
import random

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_features):
        super(Dataset, self).__init__()
        self.dataset = self.flat_dataset(dataset)
        self.video_features = video_features
        

    def __getitem__(self, index):
        record = self.dataset[index]
        vid = record['vid']
        video_feature = self.video_features[vid]
        if record['is_match'] == 1:
            s_ind, e_ind = int(record['s_ind']), int(record['e_ind'])
            #s_ind, e_ind = 1,10
        else:
            s_ind, e_ind = 1,10
        return record, video_feature, record['w_ids'], record['c_ids'], s_ind, e_ind, record['is_match']

    def __len__(self):
        return len(self.dataset)
    
    def flat_dataset(self, dataset):
        out_data = []        
        for vid in dataset:
            curr_pool = dataset[vid]
            random.shuffle(curr_pool)
            for ele in curr_pool:
                if ele['is_match'] == 0 or int(ele['s_ind']) >= int(ele['e_ind']):
                    continue
                out_data.append(ele)
        return out_data


def train_collate_fn(data):
    records, video_features, word_ids, char_ids, s_ind, e_ind, labels = zip(*data)
    word_ids, w_lens = pad_seq(word_ids)
    word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
    w_lens = np.asarray(w_lens, dtype=np.int32)  # (batch_size, )
    # process char ids
    char_ids, _ = pad_char_seq(char_ids)
    char_ids = np.asarray(char_ids, dtype=np.int32)  # (batch_size, w_seq_len, c_seq_len)
    # process video features
    vfeats, vfeat_lens = pad_video_seq(video_features)
    vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
    # process labels
  
    # convert to torch tensor
    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)

    s_labels = torch.tensor(s_ind, dtype=torch.int64)
    e_labels = torch.tensor(e_ind, dtype=torch.int64)
    labels = torch.tensor(labels, dtype=torch.int64)
    word_ids = torch.tensor(word_ids, dtype=torch.int64)
    char_ids = torch.tensor(char_ids, dtype=torch.int64)
    w_lens = torch.tensor(w_lens, dtype=torch.int64)
    return records, vfeats, vfeat_lens, w_lens, word_ids, char_ids, s_labels, e_labels, labels

def get_train_loader(dataset, video_features, configs):
    train_set = Dataset(dataset=dataset, video_features=video_features)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=configs.batch_size, shuffle=True,
                                               collate_fn=train_collate_fn)
    return train_loader


def get_test_loader(dataset, video_features, configs):
    test_set = Dataset(dataset=dataset, video_features=video_features)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=configs.batch_size, shuffle=False,
                                              collate_fn=train_collate_fn)
    return test_loader
