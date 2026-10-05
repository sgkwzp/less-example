import os
import tqdm
import json
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from scipy.ndimage import maximum_filter

import networkx as nx
from networkx.algorithms.dag import lexicographical_topological_sort

from models.models import ASDiffusionBackbone

from datasets.gtg_dataset_loader import get_data_dict, VideoDataset

from utils.metrics import Video, Checkpoint, omission_detection
from utils.utils import draw_pred, create_image_grid

from dp.graph_utils import compute_generalized_metadag_costs, generalized_metadag2vid

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


def mode_filter(x, window_size=30):
    assert window_size >= 1, "Window size must be at least 1"
    assert isinstance(window_size, int), "Window size must be an integer"
    
    n = len(x)
    filtered = np.zeros_like(x, dtype=int)
    
    for i in range(n):
        start = max(0, i - window_size // 2 + 1)
        end = min(n, i + window_size // 2)
        filtered[i] = np.bincount(x[start:end]).argmax()
    
    return filtered

def create_log_folder(dirname, naming, dataset_name, ckpt_dir):
    now = datetime.now()
    current_time = now.strftime("%m_%d_%H_%M_%S")
    
    if dirname == "debug":
        save_dir = os.path.join(ckpt_dir, naming, dataset_name, "debug")
    else:
        save_dir = os.path.join(ckpt_dir, naming, dataset_name, dirname + "_" + current_time)

        os.makedirs(os.path.join(ckpt_dir, naming, dataset_name), exist_ok=True)
        
        os.makedirs(save_dir, exist_ok=True)
    
    
    os.makedirs(os.path.join("runs/", naming, dataset_name), exist_ok=True)

    if 'debug' in dirname:
        writer = SummaryWriter(os.path.join("runs/", naming, dataset_name, "debug"))
    else: 
        writer = SummaryWriter(os.path.join("runs/", naming, dataset_name, dirname + "_" + current_time))

    return save_dir, writer

def segments_to_framewise(segments, action_list, length, default_cls=0):
    preds = torch.zeros((length)) + default_cls
    for j in range(len(segments)):
        st = int(segments[j, 0].item())
        ed = int(segments[j, 1].item())
        if st != ed:
            preds[st:ed] = action_list[j]

    return preds.long()

def get_datasets(all_params, num_classes, action2idx, actiontype2idx, is_eval):
    root_data_dir = all_params['root_data_dir']
    dataset_name = all_params['dataset_name']
    naming = all_params['naming']

    suffix = ""
    if naming == "CaptainCook4D":
        addition_name = "Other"
        suffix = "_360p"
    elif naming == "EgoPER": 
        addition_name = "Error_Addition"

    v_feature_dir = os.path.join(root_data_dir, dataset_name, all_params["v_feat_path"])
    label_dir = os.path.join(root_data_dir, dataset_name, all_params['label_path'])

    with open(os.path.join(root_data_dir, dataset_name, all_params['train_split']+".txt"), 'r') as fp:
        lines = fp.readlines()
        train_video_list = [line.strip('\n') for line in lines]

    strict_training = all_params.get('no_test_selection', False) and not is_eval
    if strict_training:
        val_video_list, test_video_list = [], []
    else:
        with open(os.path.join(root_data_dir, dataset_name, all_params['val_split']+".txt"), 'r') as fp:
            val_video_list = [line.strip() for line in fp if line.strip()]
        with open(os.path.join(root_data_dir, dataset_name, all_params['test_split']+".txt"), 'r') as fp:
            test_video_list = [line.strip() for line in fp if line.strip()]

    train_data_dict = get_data_dict(
        v_feature_dir=v_feature_dir,
        label_dir=label_dir, 
        video_list=train_video_list, 
        action2idx=action2idx, 
        actiontype2idx=actiontype2idx,
        addition_name=addition_name,
        suffix=suffix
    )

    val_data_dict = get_data_dict(
        v_feature_dir=v_feature_dir,
        label_dir=label_dir, 
        video_list=val_video_list, 
        action2idx=action2idx,
        actiontype2idx=actiontype2idx,
        addition_name=addition_name,
        suffix=suffix
    )

    test_data_dict = get_data_dict(
        v_feature_dir=v_feature_dir,
        label_dir=label_dir, 
        video_list=test_video_list, 
        action2idx=action2idx, 
        actiontype2idx=actiontype2idx,
        addition_name=addition_name,
        suffix=suffix
    )


    train_dataset = VideoDataset(root_data_dir, train_data_dict, num_classes, mode='train', naming=naming, dataset_name=dataset_name)
    val_dataset = VideoDataset(root_data_dir, val_data_dict, num_classes, mode='test', naming=naming, dataset_name=dataset_name)
    test_dataset = VideoDataset(root_data_dir, test_data_dict, num_classes, mode='test', naming=naming, dataset_name=dataset_name)
    
    return {"train": train_dataset, "val": val_dataset, "test": test_dataset}

class Runner:
    def __init__(self, args):
        all_params = json.load(open(args.config))
        seed = int(all_params.get('seed', 0))
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        all_params['no_test_selection'] = getattr(args, 'no_test_selection', False)
        root_data_dir = all_params['root_data_dir']
        dataset_name = all_params['dataset_name']
        input_dim = all_params['input_dim']
        self.naming = all_params['naming']
        self.lr = all_params['learning_rate']
        self.weight_decay = all_params['weight_decay']
        self.num_epochs = all_params['num_epochs']
        self.log_freq = all_params['log_freq']
        self.ignore_idx = all_params['ignore_idx']
        self.batch_size = all_params['batch_size']
        self.num_iterations = all_params['num_iterations']
        self.ckpt_dir = all_params["ckpt_dir"]
        self.drop_base = all_params["drop_base"]
        self.is_vis = args.vis
        self.save_scores_dir = getattr(args, 'save_scores', None)
        self.is_training = not args.eval
        # Strict few-shot protocol: never inspect the test loader during training
        # and always export the final checkpoint.  The historical code selected
        # ``best_checkpoint.pth`` from test-set F1 after every epoch.
        self.no_test_selection = getattr(args, "no_test_selection", False)
        self.dataset_name = dataset_name
        
        with open(os.path.join(root_data_dir, 'action2idx.json'), 'r') as fp:
            self.action2idx = json.load(fp)[dataset_name]
        
        with open(os.path.join(root_data_dir, 'actiontype2idx.json'), 'r') as fp:
            self.actiontype2idx = json.load(fp)

        with open(os.path.join(root_data_dir, 'idx2action.json'), 'r') as fp:
            self.idx2action = json.load(fp)[dataset_name]
        
        with open(os.path.join(root_data_dir, 'idx2actiontype.json'), 'r') as fp:
            self.idx2actiontype = json.load(fp)

        self.num_classes = len(self.action2idx)
        self.gtg_num_classes =  self.num_classes * 2
        self.bg_idx = self.action2idx["BG"]
        
        # error as additional class
        self.draw_idx2action = dict(self.idx2action)
        self.draw_idx2action["-1"] = "Error"

        # last type always addition
        # if addition in actiontype2idx, then minus 1
        if self.naming == "EgoPER":
            self.addition_idx = self.actiontype2idx["Error_Addition"]
        elif self.naming == "CaptainCook4D":
            self.addition_idx = self.actiontype2idx["Preparation Error"]
        self.num_types = len(self.actiontype2idx) - 1
        # self.num_types = self.actiontype2idx[self.addition_name] #len(self.actiontype2idx) - 1 # process addition prototypes independently

        if "node_drop_base" in all_params:
            self.node_drop_base = all_params["node_drop_base"]
        else:
            self.node_drop_base = self.gtg_num_classes
        
        if "window_size" in all_params:
            self.window_size = all_params["window_size"]
        else:
            self.window_size = 30
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ASDiffusionBackbone(input_dim, self.gtg_num_classes, self.num_classes, self.num_types, self.addition_idx, self.device, all_params['background_weight'])
        self.model.to(self.device)

        self.opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        dataset_dict = get_datasets(all_params, self.num_classes, self.action2idx, self.actiontype2idx, args.eval)


        loader_workers = 0
        if args.eval:
            self.test_loader = torch.utils.data.DataLoader(dataset_dict['test'], batch_size=1, shuffle=False, num_workers=loader_workers)
            self.val_loader = torch.utils.data.DataLoader(dataset_dict['val'], batch_size=1, shuffle=False, num_workers=loader_workers)
            self.train_loader = torch.utils.data.DataLoader(dataset_dict['train'], batch_size=1, shuffle=False, num_workers=loader_workers)
            checkpoint_name = "last_checkpoint.pth" if self.no_test_selection else "best_checkpoint.pth"
            self.load_dir = os.path.join(self.ckpt_dir, self.naming, dataset_name, args.dir, checkpoint_name)
            self.save_dir = os.path.join(self.ckpt_dir, self.naming, dataset_name, args.dir)
            self.writer = None
        else:
            self.test_loader = torch.utils.data.DataLoader(dataset_dict['test'], batch_size=1, shuffle=False, num_workers=loader_workers)
            self.val_loader = torch.utils.data.DataLoader(dataset_dict['val'], batch_size=1, shuffle=False, num_workers=loader_workers)
            self.train_loader = torch.utils.data.DataLoader(dataset_dict['train'], batch_size=1, shuffle=True, num_workers=loader_workers)
            self.save_dir, self.writer = create_log_folder(args.dir, self.naming, dataset_name, self.ckpt_dir) 
            if getattr(args, 'exact_dir', False):
                if self.writer is not None:
                    self.writer.close()
                self.save_dir = os.path.join(self.ckpt_dir, self.naming, dataset_name, args.dir)
                os.makedirs(self.save_dir, exist_ok=True)
                self.writer = SummaryWriter(os.path.join(self.save_dir, 'logs'))
        
        self.G = dataset_dict['train'].G

        ############################################
        # generate error/normal description features
        with open(os.path.join(root_data_dir, all_params['dataset_name'], all_params['simple_error_filename']+".txt"), "r") as fp:
            self.error_list = fp.readlines()

        normal_filename = all_params.get("normal_action_filename", "normal_actions")
        normal_path = os.path.join(root_data_dir, all_params['dataset_name'], normal_filename + ".txt")
        if not os.path.exists(normal_path):
            # Released EgoPER configs store this file as ``<recipe>_normal_actions``.
            normal_path = os.path.join(root_data_dir, all_params['dataset_name'], f"{all_params['dataset_name']}_normal_actions.txt")
        with open(normal_path, "r") as fp:
            self.normal_action_list = fp.readlines()
        

        self.action_error_dict = {}
        for error in self.error_list:
            name, des = error.split(" ")
            _, action, _, action_type, err_idx = name.split("_")
            action = int(action)
            action_type = int(action_type)

            feature = np.load(os.path.join(root_data_dir, all_params['dataset_name'], all_params['simple_error_path'], name+".npy"))
            feature = torch.from_numpy(feature).float()

            if action not in self.action_error_dict:
                self.action_error_dict[action] = {}
            
            if action_type not in self.action_error_dict[action]:
                self.action_error_dict[action][action_type] = []

            self.action_error_dict[action][action_type].append(feature)

        # average all the error feature
        self.merge_action_error_dict = {}
        for k, v in self.action_error_dict.items():
            self.merge_action_error_dict[k] = {}
            for s_k, s_v in self.action_error_dict[k].items():
                self.merge_action_error_dict[k][s_k] = torch.stack(s_v).mean(0)

        
        self.normal_action_dict = {}
        for normal_action in self.normal_action_list:
            name, des = normal_action.split(" ")
            _, action = name.split("_")

            feature = np.load(os.path.join(root_data_dir, all_params['dataset_name'], "vc_normal_action_features", name+".npy"))
            feature = torch.from_numpy(feature).float()

            if action not in self.normal_action_dict:
                self.normal_action_dict[int(action)] = feature
        ############################################

        ################################################
        # for evaluation, ignore non-exsting action type
        ################################################
        if self.naming == "EgoPER":
            self.ignore_actions = []
        elif self.naming == "CaptainCook4D":
            # self.ignore_actions = [0, 6] # normal and other
            self.ignore_actions = ["Normal", "Other"]

        exist_type = []
        if self.no_test_selection:
            # Strict few-shot mode must not inspect test labels merely to
            # build the error-recognition ignore list.  Keep every configured
            # error type and let evaluation handle absent types naturally.
            exist_type = list(range(1, self.num_types + 1))
        else:
            for video_idx, data in enumerate(self.test_loader):
                v_feature, label, type_label, video  = data
                type_label = type_label.squeeze(0)
                for idx in range(self.num_types): # 0 is normal
                    action_type = idx + 1
                    if action_type not in exist_type and (type_label == action_type).sum() > 0: # if the type exists
                        exist_type.append(action_type)
        for action_type in range(self.num_types+1): # 0 is normal
            # if action_type not in exist_type and action_type not in self.ignore_actions:
            if action_type not in exist_type and self.idx2actiontype[str(action_type)] not in self.ignore_actions:
                # self.ignore_actions.append(action_type)
                self.ignore_actions.append(self.idx2actiontype[str(action_type)])
        
        print("Ignore specific or non-exsting action type for error recognition:", self.ignore_actions)
        
    def from_framewise_to_steps(self, labels, ignore_bg=True):
    
        pre_label = None

        steps = []
        timestamps = []
        st, ed = 0, 0
        
        for i in range(len(labels)):
            label = labels[i].unsqueeze(0)

            if pre_label is None:
                pre_label = label
            
            if pre_label != label:
                if not ignore_bg or (ignore_bg and pre_label != self.bg_idx):
                    steps.append(pre_label)
                    timestamps.append([st, ed])
                st = ed
                pre_label = label

            ed += 1
        if not ignore_bg or (ignore_bg and pre_label != self.bg_idx):
            steps.append(pre_label)
            timestamps.append([st, ed])

        return torch.cat(steps, 0).long(), torch.tensor(timestamps).float()

    def get_data_sample(self):
        v_feature, label, type_label, video = next(iter(self.train_loader))
        v_feature, label, type_label = v_feature.to(self.device), label.to(self.device), type_label.to(self.device)
        return v_feature, label, type_label, video
            
    def relabel_bg(self, steps):
        for i in range(len(steps) - 1):
            if steps[i] == self.bg_idx:
                if steps[i+1] == self.bg_idx:
                    print("action after bg should a normal action")
                    exit(0)
                else:
                    steps[i] = self.num_classes - 1 + steps[i+1]
        return steps

    def compute_per_out_log(self, joint_per_metrics, use_ignore=False, mode="as"):
        per_out_log = []
        num_thresholds = 0
        avg_f1_thresholds = 0
        if mode == "as":
            col1 = 10
        else:
            col1 = 20
        for t, _ in joint_per_metrics.items():
            f1_list = []
            # per_out_log.append("|\tAction\t|\tPrecision@%s\t|\tRecall@%s\t|\tF1@%s\t|\n"%(t, t, t))
            per_out_log.append(f"|{'Action':^{col1}}|{'Precision@%s'%t:^{15}}|{'Recall@%s'%t:^{15}}|{'F1@%s'%t:^{10}}|\n")
            

            total_tp, total_fp, total_fn = 0, 0, 0
            theta = 0
            for action, tp_fp_fn in joint_per_metrics[t].items():
                tp, fp, fn = 0, 0, 0
                if mode == "er":
                    action = self.idx2actiontype[str(action)]
                for i in range(len(tp_fp_fn[0])):
                    tp += tp_fp_fn[0][i]
                    fp += tp_fp_fn[1][i]
                    fn += tp_fp_fn[2][i]
                    if action not in self.ignore_actions:
                        total_tp += tp_fp_fn[0][i]
                        total_fp += tp_fp_fn[1][i]
                        total_fn += tp_fp_fn[2][i]
                
                p = tp / float(tp+fp)
                r = tp / float(tp+fn)
                if np.isnan(p):
                    p = 0.0
                if np.isnan(r):
                    r = 0.0
                if p+r == 0:
                    f1 = 0.0
                else:
                    f1 = 2.0 * (p*r) / (p+r)
                f1 = np.nan_to_num(f1)
                p = p * 100
                r = r * 100
                f1 = f1 * 100
                if mode == "ed":
                    action = "Error" if action == 1 else "Normal"
                    # out_log = "|\t%s\t|\t%.1f\t|\t%.1f\t|\t%.1f\t|\n"%(action, p, r, f1)
                out_log = f"|{action:^{col1}}|{p:^{15}.1f}|{r:^{15}.1f}|{f1:^{10}.1f}|\n"
                # print(action, "precision:", np.mean(np.array(p_r[0])), "recall:", np.mean(np.array(p_r[1])))
                per_out_log.append(out_log)
                if f1 != 0 and action not in self.ignore_actions:
                    theta += 1

                if not use_ignore or action not in self.ignore_actions:
                    f1_list.append(f1)
            
            
            per_out_log.append("\n")
            if use_ignore:
                per_out_log.append("Ignore action: ")
                for exclude_action in self.ignore_actions:
                    per_out_log.append(exclude_action+", ")
            per_out_log.append("\n")
            per_out_log.append("Avg F1@%s: %.1f\n\n"%(t, np.array(f1_list).mean()))
            
            # per_out_log.append("\n")
            # per_out_log.append("|")
            # for f1 in f1_list:
            #     per_out_log.append("%.1f|"%(f1))
            # per_out_log.append("\n\n")

            avg_f1_thresholds += np.array(f1_list).mean()

            if mode == "er":
                total_p = total_tp / float(total_tp+total_fp)
                total_r = total_tp / float(total_tp+total_fn)
                if np.isnan(total_p):
                    total_p = 0.0
                if np.isnan(total_r):
                    total_r = 0.0
                if total_p+total_r == 0:
                    total_f1 = 0.0
                else:
                    total_f1 = 2.0 * (total_p*total_r) / (total_p+total_r)
                total_f1 = np.nan_to_num(total_f1)
                total_p = total_p * 100
                total_r = total_r * 100
                total_f1 = total_f1 * 100
                # print(len(self.actiontype2idx), len(self.ignore_actions), theta)
                # per_out_log.append("|Total|Total P|Total R|Total F1|Total w F1|theta|\n")
                # out_log = "|All (excluding normal type or BG)|%.1f|%.1f|%.1f|%.1f|%d|\n\n"%(total_p, total_r, total_f1, total_f1/(len(self.actiontype2idx) - len(self.ignore_actions) - theta + 1), theta)
                # print(action, "precision:", np.mean(np.array(p_r[0])), "recall:", np.mean(np.array(p_r[1])))
                
                eacc = theta / (len(self.actiontype2idx) - len(self.ignore_actions))
                total_w_f1 = total_f1 * eacc
                per_out_log.append(f"|{'All Precision':^{15}}|{'All Recall':^{15}}|{'All F1':^{15}}|{'All w-F1@%s'%t:^{15}}|{'EAcc':^{15}}|\n")
                per_out_log.append(f"|{total_p:^{15}.1f}|{total_r:^{15}.1f}|{total_f1:^{15}.1f}|{total_w_f1:^{15}.1f}|{eacc*100:^{15}.1f}|\n\n")
                # per_out_log.append(out_log)

        return per_out_log, avg_f1_thresholds / len(joint_per_metrics)

    def erm(self, pred, no_drop_pred, feature):
        cosine_similarity = nn.CosineSimilarity()

        type_pred = np.zeros((len(pred)))
        type_prob = np.zeros((self.num_types + 1, len(pred)))

        target_area = pred == -1

        for i in range(len(target_area)):
            # if is target frame (error)
            if target_area[i]:
                target_action = no_drop_pred[i]
                target_frame_feature = feature[i]
                best_sim = None
                target_type = 0

                # if action in no_drop_pred is background, label remain -1
                if target_action == self.bg_idx:
                    type_prob[self.addition_idx][i] = 1.0
                else:
                    for k, v in self.merge_action_error_dict[target_action].items():
                        sim = torch.sigmoid((v * self.normal_action_dict[target_action] * target_frame_feature).sum() * 200)
                        if best_sim is None or sim > best_sim:
                            best_sim = sim
                            target_type = k
                    type_prob[target_type][i] = best_sim
            else:
                type_prob[0][i] = 1.0

        ## smoothing
        smoothed_output = np.zeros_like(type_prob)
        for c in range(type_prob.shape[0]):
            smoothed_output[c] = maximum_filter(type_prob[c], size=self.window_size)
        type_prob = smoothed_output

        type_pred = np.argmax(type_prob, 0)
        error_pred = np.copy(type_pred)
        error_pred[error_pred > 0] = 1

        # The signed margin is a continuous surrogate for ED.  At decision
        # point zero, ``margin > 0`` must reproduce ``error_pred``; the eval
        # exporter checks this elementwise before allowing an AUC comparison.
        error_margin = np.max(type_prob[1:], axis=0) - type_prob[0]
        return pred, type_pred, error_pred, error_margin

    def train(self):
        global_step = 0
        best_score = 0.0
        
        for epoch in range(self.num_epochs):
            
            samples = []

            for idx in range(self.num_iterations):
                feature, label, type_label, video = self.get_data_sample()

                video = video
                label = label.squeeze(0).long().cpu()
                type_label = type_label.squeeze(0).long().cpu()

                action_logits, frame_features = self.model(feature.permute(0, 2, 1), label)

                steps, timestamps = self.from_framewise_to_steps(label, ignore_bg=False)

                # relabel background segment
                new_steps = self.relabel_bg(steps.clone())
                new_label = segments_to_framewise(timestamps, new_steps, feature.size(2))

                sample = {}
                sample["action_logits"] = action_logits.squeeze(0)
                sample["frame_features"] = frame_features.squeeze(0).permute(1, 0)
                sample["org_frame_features"] = feature.squeeze(0).permute(1, 0)
                sample["normal_action_features"] = self.normal_action_dict
                sample["framewise_labels"] = new_label
                sample["framewise_type_labels"] = type_label
                sample["video_id"] = video
                sample["metagraph"] = self.G.graph_info["metagraph"]
                sample["gmetagraph"] = self.G.graph_info["gmetagraph"]
                
                samples.append(sample)
                
                if (idx + 1) % (self.batch_size) == 0 or (idx + 1) == self.num_iterations:
                    # update networks
                    gtg2vid_loss = self.model.gtg2vid_loss(samples)
                    out_log = ""
                    total_loss = 0
                    self.opt.zero_grad()
                    for k, v in gtg2vid_loss.items():
                        out_log += "%s:%.3f\t"%(k, v.item())
                        total_loss += v
                    total_loss.backward()
                    self.opt.step()
                    
                    samples = []
                    global_step += 1
                    if (idx + 1) % self.log_freq == 0:
                        print('Epoch: [%d/%d], Iter: [%d/%d]'%(epoch+1, self.num_epochs, idx+1, self.num_iterations), out_log)
                    if self.writer is not None:
                        self.writer.add_scalar("Loss/train", total_loss, global_step)
                    

            if self.no_test_selection:
                # Do not call evaluate(): it iterates over ``self.test_loader``.
                # The final epoch is the pre-registered checkpoint in G0/G1.
                torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': self.model.state_dict()
                            },
                            os.path.join(self.save_dir, "last_checkpoint.pth"))
            else:
                f1_vid = self.evaluate(global_step)
                if f1_vid >= best_score:
                    best_score = f1_vid
                    print('Save best weight at epoch: %d'%(epoch + 1))
                    torch.save({
                                'epoch': epoch + 1,
                                'model_state_dict': self.model.state_dict()
                                },
                                os.path.join(self.save_dir, "best_checkpoint.pth"))
                
            if self.writer is not None:
                self.writer.flush()

    def evaluate(self, global_step=None):
        
        if not self.is_training:
            checkpoint = torch.load(self.load_dir)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Load from epoch %d"%(checkpoint['epoch']))
            loaders = {"test": self.test_loader}
        else:
            loaders = {"test": self.test_loader}

        self.model.eval()
        
        with torch.no_grad():

            for loader_name, loader in loaders.items():
                video_pair_list = []
                type_video_pair_list = []
                error_video_pair_list = []
                predstep_steps = []
                acc_list = []
                tpr_list = []
                fpr_list = []

                for video_idx, data in enumerate(loader):

                    v_feature, label, type_label, video  = data
                    video = video[0]
                    feature = v_feature.to(self.device)

                    action_logits, frame_features = self.model(feature.permute(0, 2, 1))

                    label = label.squeeze(0)
                    type_label = type_label.squeeze(0)
                    error_label = type_label.clone()

                    sample = {}
                    sample["action_logits"] = action_logits.squeeze(0).cpu()
                    sample["label"] = label
                    sample["num_classes"] = self.num_classes
                    sample["video_id"] = video
                    sample["metagraph"] = self.G.graph_info["metagraph"]
                    sample["gmetagraph"] = self.G.graph_info["gmetagraph"]

                    gmetadag = sample["gmetagraph"]
                    sorted_node_ids = list(lexicographical_topological_sort(gmetadag))
                    idx2node = {idx: node_id for idx, node_id in enumerate(sorted_node_ids)}
                    zx_costs, drop_costs, node_drop_costs = compute_generalized_metadag_costs(sample, idx2node, self.drop_base, self.node_drop_base)
                    _, pred, type_pred = generalized_metadag2vid(zx_costs.cpu().numpy(), drop_costs.cpu().numpy(), node_drop_costs.cpu().numpy(), gmetadag, idx2node)

                    # convert all background into the same class
                    pred[pred >= self.num_classes] = self.bg_idx

                    ######### second round, no drop 
                    zx_costs, drop_costs, node_drop_costs = compute_generalized_metadag_costs(sample, idx2node, -100, -200)
                    _, no_drop_pred, _ = generalized_metadag2vid(zx_costs.cpu().numpy(), drop_costs.cpu().numpy(), node_drop_costs.cpu().numpy(), gmetadag, idx2node)

                    # convert all background into the same class
                    no_drop_pred[no_drop_pred >= self.num_classes] = self.bg_idx

                    pred, type_pred, error_pred, error_margin = self.erm(pred, no_drop_pred, feature.permute(0, 2, 1).cpu().squeeze(0))

                    if self.save_scores_dir:
                        os.makedirs(self.save_scores_dir, exist_ok=True)
                        labels_np = (error_label.cpu().numpy().reshape(-1) > 0).astype(np.int64)
                        scores_np = np.asarray(error_margin, dtype=np.float64).reshape(-1)
                        if labels_np.shape != scores_np.shape:
                            raise ValueError(f'GTG score/label length mismatch for {video}: {scores_np.shape} vs {labels_np.shape}')
                        np.savez_compressed(os.path.join(self.save_scores_dir, f'{video}.npz'), labels=labels_np, scores=scores_np, fps=np.array([10], dtype=np.int64), video_id=np.array([video]))



                    ###############################
                    # convert -1 (addition prediction) to background for computing action segmentation performance
                    ## 03052025, count error as additional class
                    # pred[pred == -1] = self.bg_idx
                    # label[label == -1] = self.bg_idx
                    ###############################

                    # let error as an additional class
                    label_w_error_cls = label.clone()
                    label_w_error_cls[type_label != 0] = -1

                    vis_dir = 'vis'
                    log_dir = 'log'              

                    # pred = mode_filter(pred, windo_size=20)

                    if self.is_vis:
                        if not os.path.exists(os.path.join(self.save_dir, vis_dir)):
                            os.mkdir(os.path.join(self.save_dir, vis_dir))
                        if not os.path.exists(os.path.join(self.save_dir, vis_dir, 'gt')):
                            os.mkdir(os.path.join(self.save_dir, vis_dir, 'gt'))

                        draw_pred(label_w_error_cls.long(), "gt", self.draw_idx2action, os.path.join(self.save_dir, vis_dir, "gt", video+"_as"))
                        
                        # only for visualization
                        # print(pred.shape)
                        # pred = median_filter(pred, size=20)
                        # pred = mode_filter(pred, window_size=20)
                        draw_pred(torch.from_numpy(pred).long(), "as", self.draw_idx2action, os.path.join(self.save_dir, vis_dir, video+"_as"))
                        draw_pred(type_label.squeeze(0).cpu().long(), "gt", self.idx2actiontype, os.path.join(self.save_dir, vis_dir, "gt", video+'_er'))
                        draw_pred(torch.from_numpy(type_pred).long(), "er", self.idx2actiontype, os.path.join(self.save_dir, vis_dir, video+'_er'))
                        draw_pred(error_label.squeeze(0).cpu().long(), "gt", self.idx2actiontype, os.path.join(self.save_dir, vis_dir, "gt", video+'_ed'))
                        draw_pred(torch.from_numpy(error_pred).long(), "ed", self.idx2actiontype, os.path.join(self.save_dir, vis_dir, video+'_ed'))
                        
                        ######################### save output
                        if not os.path.exists(os.path.join(self.save_dir, "output")):
                            os.mkdir(os.path.join(self.save_dir, "output"))
                        if not os.path.exists(os.path.join(self.save_dir, "output", "tas_nodrop")):
                            os.mkdir(os.path.join(self.save_dir, "output", "tas_nodrop"))
                        if not os.path.exists(os.path.join(self.save_dir, "output", "tas")):
                            os.mkdir(os.path.join(self.save_dir, "output", "tas"))
                        if not os.path.exists(os.path.join(self.save_dir, "output", "ed")):
                            os.mkdir(os.path.join(self.save_dir, "output", "ed"))
                        if not os.path.exists(os.path.join(self.save_dir, "output", "er")):
                            os.mkdir(os.path.join(self.save_dir, "output", "er"))
                        
                        with open(os.path.join(self.save_dir, "output", "tas_nodrop", video+".txt"), "w") as fp:
                            json.dump(no_drop_pred.tolist(), fp)
                        with open(os.path.join(self.save_dir, "output", "tas", video+".txt"), "w") as fp:
                            json.dump(pred.tolist(), fp)
                        with open(os.path.join(self.save_dir, "output", "ed", video+".txt"), "w") as fp:
                            json.dump(error_pred.tolist(), fp)
                        with open(os.path.join(self.save_dir, "output", "er", video+".txt"), "w") as fp:
                            json.dump(type_pred.tolist(), fp)
                        ##########################

                    # video_pair_list.append(Video(video_idx, pred.tolist(), label.tolist()))
                    video_pair_list.append(Video(video_idx, pred.tolist(), label_w_error_cls.tolist()))

                    type_video_pair_list.append(Video(video_idx, type_pred.tolist(), type_label.cpu().tolist()))

                    # assign all errors into class 1
                    error_label[error_label > 0] = 1
                    error_video_pair_list.append(Video(video_idx, error_pred.tolist(), error_label.tolist()))

                    # for omission detection, ignore errors
                    pred[pred == -1] = self.bg_idx
                    label[label == -1] = self.bg_idx
                    steps, _ = self.from_framewise_to_steps(label, ignore_bg=True)
                    pred_steps, _ = self.from_framewise_to_steps(torch.tensor(pred).long(), ignore_bg=True)
                    predstep_steps.append([pred_steps, steps])

                    

                if self.naming == "EgoPER":
                    oIoU, oAcc = omission_detection(self.G.graph_info["graph"], predstep_steps)
                    # print("|oIoU:%.1f|oAcc:%.1f|"%(oIoU*100, oAcc*100))
                    omit_log = []
                    omit_log.append("Omission Detecion:\n")
                    omit_log.append("|oIoU:%.1f|oAcc:%.1f|"%(oIoU*100, oAcc*100))
                else:
                    omit_log = None

                # ignore addition error while computing action segmentation
                ckpt = Checkpoint(bg_class=[self.ignore_idx])
                ckpt.add_videos(video_pair_list)
                as_out, as_per_out = ckpt.compute_metrics()
                # as_out_log = "|Action Segmentation|Edit:%.1f|Acc:%.1f|F1@.1:%.1f|F1@.25:%.1f|F1@.5:%.1f|"%(as_out['edit']*100, as_out['acc']*100, as_out['F1@0.100']*100, as_out['F1@0.250']*100, as_out['F1@0.500']*100)
                as_out_log = "|Edit:%.1f|Acc:%.1f|"%(as_out['edit']*100, as_out['acc']*100)
                as_per_out_log, as_avg_f1 = self.compute_per_out_log(as_per_out, mode='as')
                # print(as_out_log)

                ckpt = Checkpoint(bg_class=[self.ignore_idx])
                # ckpt = Checkpoint(bg_class=[0]) # ignore normal type
                ckpt.add_videos(type_video_pair_list)
                er_out, er_per_out = ckpt.compute_metrics()
                er_out_log = "|Error Recognition|F1@.1:%.1f|F1@.25:%.1f|F1@.5:%.1f|"%(er_out['F1@0.100']*100, er_out['F1@0.250']*100, er_out['F1@0.500']*100)
                er_per_out_log, er_avg_f1 = self.compute_per_out_log(er_per_out, use_ignore=True, mode='er')
                # print(er_out_log)

                ckpt = Checkpoint(bg_class=[self.ignore_idx])
                ckpt.add_videos(error_video_pair_list)
                ed_out, ed_per_out = ckpt.compute_metrics()
                ed_out_log = "|Error Detection|F1@.1:%.1f|F1@.25:%.1f|F1@.5:%.1f|"%(ed_out['F1@0.100']*100, ed_out['F1@0.250']*100, ed_out['F1@0.500']*100)
                ed_per_out_log, ed_avg_f1 = self.compute_per_out_log(ed_per_out, mode='ed')
                # print(ed_out_log)


                if not os.path.exists(os.path.join(self.save_dir, log_dir)):
                    os.mkdir(os.path.join(self.save_dir, log_dir))
                
                # as_out_logs = [
                #     "|Mode|Split|Domain|Edit|Acc|F1@0.100|F1@0.250|F1@0.500|\n",
                #     "|---|---|---|---|---|---|---|---|\n",
                #     as_out_log+"\n\n"
                # ]
                # as_out_logs = [as_out_log+"\n\n"]
                # as_out_logs.extend(as_per_out_log)

                as_out_logs = as_per_out_log
                as_out_logs.append("\n\n")
                as_out_logs.append(as_out_log)

                # er_out_logs = [
                #     "|Mode|Split|Domain|Edit|Acc|F1@0.100|F1@0.250|F1@0.500|\n",
                #     "|---|---|---|---|---|---|---|---|\n",
                #     er_out_log+"\n\n"
                # ]
                # er_out_logs.extend(er_per_out_log)
                # er_out_logs = [er_out_log+"\n\n"]
                # er_out_logs.extend(er_per_out_log)
                er_out_logs = er_per_out_log

                # ed_out_logs = [
                #     "|Mode|Split|Domain|Edit|Acc|F1@0.100|F1@0.250|F1@0.500|EDA|\n",
                #     "|---|---|---|---|---|---|---|---|---|\n",
                #     ed_out_log+"\n\n"
                # ]
                # ed_out_logs.extend(ed_per_out_log)
                # ed_out_logs = [ed_out_log+"\n\n"]
                # ed_out_logs.extend(ed_per_out_log)
                ed_out_logs = ed_per_out_log
                ed_out_logs.append("\n\n")
                
                if omit_log is not None:
                    ed_out_logs.extend(omit_log)


                # final_acc = np.array(acc_list).mean()
                # final_tpr = np.array(tpr_list).mean()
                # final_fpr = np.array(fpr_list).mean()

                # acc_tpr_fpr_log = ["\n%.1f, %.1f, %.1f\n"%(final_acc*100, final_tpr*100, final_fpr*100)]
                # ed_out_logs.extend(acc_tpr_fpr_log)

                with open(os.path.join(self.save_dir, log_dir, 'action_segmentation.txt'), "w") as fp:
                    fp.writelines(as_out_logs)
                with open(os.path.join(self.save_dir, log_dir, 'error_recognition.txt'), "w") as fp:
                    fp.writelines(er_out_logs)
                with open(os.path.join(self.save_dir, log_dir, 'error_detection.txt'), "w") as fp:
                    fp.writelines(ed_out_logs)

                if self.writer is not None:
                    self.writer.add_scalar("AS_F1@0.500/valid", as_out['F1@0.500']*100, global_step)
                    self.writer.add_scalar("ER_F1@0.500/valid", er_out['F1@0.500']*100, global_step)
                    self.writer.add_scalar("ED_F1@0.500/valid", ed_out['F1@0.500']*100, global_step)
                    self.writer.add_scalar("AVG_ER_F1/valid", er_avg_f1, global_step)
                    self.writer.add_scalar("AVG_ED_F1/valid", ed_avg_f1, global_step)
                    if self.is_vis:
                        grid = create_image_grid(os.path.join(self.save_dir, vis_dir))
                        self.writer.add_image('Images/valid', grid, global_step)

            
            self.model.train() 
            print("Evalutation Done...")
            return (as_avg_f1 + ed_avg_f1 + er_avg_f1) / 3
