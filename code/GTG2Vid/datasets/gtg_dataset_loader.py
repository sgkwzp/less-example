import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets.loader_graph import GraphLoader

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

def get_data_dict(v_feature_dir, label_dir, video_list, action2idx, actiontype2idx, addition_name="Error_Addition", suffix=""):
    
        
    data_dict = {k:{
        'v_feature': None,
        'label_seq': None,
        'type_label_seq': None,
        } for k in video_list
    }
    
    print(f'Loading Dataset ...')
    
    for video in tqdm(video_list):
        v_feature_file = os.path.join(v_feature_dir, '{}.npy'.format(video+suffix))
        
        event_file = os.path.join(label_dir, '{}.txt'.format(video))

        with open(os.path.join(event_file), 'r') as fp:
            event = fp.readlines()
        
        frame_num = len(event)
                
        label_seq = np.zeros((frame_num,))
        type_label_seq = np.zeros((frame_num,))
        for i in range(frame_num):
            tokens = event[i].split('|')
            if len(tokens) == 2:
                action, action_type = tokens
            elif len(tokens) == 3:
                action, action_type, error_des = tokens
            
            action_type = action_type.strip("\n")
            
            if action_type == addition_name:
                label_seq[i] = -1
            else:
                if action in action2idx:
                    label_seq[i] = action2idx[action]
                else:
                    print(video, action_type, action, "is not in dict, therefore we assign it to %s" % addition_name)
                    label_seq[i] = -1
                    action_type = addition_name

            type_label_seq[i] = actiontype2idx[action_type]
        

        v_feature = np.load(v_feature_file)
        assert(v_feature.shape[0] == label_seq.shape[0])

        data_dict[video]['v_feature'] = torch.from_numpy(v_feature).float()
        data_dict[video]['label_seq'] = torch.from_numpy(label_seq).long()
        data_dict[video]['type_label_seq'] = torch.from_numpy(type_label_seq).long()

    return data_dict


class VideoDataset(Dataset):
    def __init__(self, root_data_dir, data_dict, class_num, mode, naming=None, dataset_name=None):
        super(VideoDataset, self).__init__()
        
        assert(mode in ['train', 'test'])
        
        self.data_dict = data_dict
        self.class_num = class_num
        self.mode = mode
        self.video_list = [i for i in self.data_dict.keys()]

        self.G = GraphLoader(naming, dataset_name, self.class_num)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        video = self.video_list[idx]
        v_feature = self.data_dict[video]['v_feature']
        label = self.data_dict[video]['label_seq']
        type_label = self.data_dict[video]['type_label_seq']
        v_feature = v_feature.T   # F x T

        return v_feature, label, type_label, video
