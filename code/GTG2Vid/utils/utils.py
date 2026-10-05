import os
import json
import random
import torch
import math
import numpy as np
from scipy import stats
try:
    import matplotlib
    from matplotlib import pyplot as plt
except ImportError:
    matplotlib = None
    plt = None
try:
    from torchvision.utils import make_grid
    from torchvision.io import read_image
except ImportError:
    make_grid = None
    read_image = None
from scipy.ndimage import generic_filter
import torch.nn.functional as F

def generate_partitions(inputs):
    cur_class = None
    start = 0
    step_partitions = []
    for i in range(len(inputs)):
        if inputs[i] != cur_class and cur_class is not None:
            step_partitions.append((cur_class, i - start + 1))
            start = i + 1
        cur_class = inputs[i]
    step_partitions.append((inputs[len(inputs) - 1], len(inputs) - start + 1))
    return step_partitions

def draw_pred(outputs, name, mapping, save_path, category_colors=None):
    clean_version = False #True
    
    if category_colors is None:
        mycmap = plt.matplotlib.cm.get_cmap('rainbow', len(mapping))
        category_colors = [matplotlib.colors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]
    
    gt_partitions = generate_partitions(outputs)
    
    plt.figure(figsize=(16, 4))
    plt.subplots_adjust(top=0.5)
    data_cum = 0
    for i, (l, w) in enumerate(gt_partitions):
        # if clean_version:
        #     # print(l, end=' ')
        #     rects = plt.barh(name, w, left=data_cum, height=0.3, color=category_colors[l.item()])
        # else:
        #     rects = plt.barh(" ", w, left=data_cum, height=0.3,
        #                     label=mapping[str(l.item())], color=category_colors[l.item()])
        #     text_color = "black" # transparent color
        #     plt.bar_label(rects, labels = [l.item()], label_type='center', color=text_color, fontsize='small')
        
        rects = plt.barh(name, w, left=data_cum, height=0.3, label=mapping[str(l.item())], color=category_colors[l.item()])
        plt.yticks([])
        data_cum += w
    
    # print()
    # if not clean_version:
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     by_label = dict(zip(labels, handles))
    #     plt.legend(by_label.values(), by_label.keys(), ncol=4, bbox_to_anchor=(1.1, 1.5), loc='upper right', fontsize='small')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), ncol=4, bbox_to_anchor=(1, 2.2), loc='upper right', fontsize='small')

    plt.savefig(save_path+'.png')
    plt.clf()
    plt.close()

def create_image_grid(img_dirs):
    img_list = []
    for img_dir in os.listdir(img_dirs):
        filenames = os.listdir(os.path.join(img_dirs, img_dir))
        for filename in filenames:
            img_list.append(read_image(os.path.join(img_dirs, img_dir, filename)))
    grid = make_grid(img_list, nrow=3)
    return grid
