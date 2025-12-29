import torch
import numpy as np
import os, json
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from sklearn.metrics import jaccard_score
import random
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image, normalize, erase

scales = ['8','14','28','56','112']

ours = [torch.load('priming_results/ours_{}.pth'.format(scale,scale)) for scale in scales]
img_path = 'hop/images'
with open('hop/images/metadata.json') as f:
    metadata = json.load(f, object_pairs_hook=OrderedDict)

ours_scores = []
random_scores = []
root = 'hop/human_maps'

for np_path in os.listdir(root):
    imgid, label = np_path.split('_')[0]+'.jpg', np_path.split('_')[1][:-4]
    label = 'mouse' if label == 'computer mouse' else label
    attn_idx = metadata[imgid]['absent_objs'].index(label)
    human_attn = torch.tensor(np.load(os.path.join(root,np_path)))[None,None,:,:]
    human_attn_downsampled = F.interpolate(human_attn, size=(30,30), mode='bilinear').squeeze()
    human_attn_downsampled = (human_attn_downsampled - human_attn_downsampled.min())/(human_attn_downsampled.max() - human_attn_downsampled.min())
    ours_attn = torch.stack([ours[i][imgid][attn_idx] for i in range(5)]).mean(0)[None,None,:,:]
    ours_attn_downsampled = F.interpolate(ours_attn, size=(30,30), mode='bilinear').squeeze() 
    ours_attn_downsampled = (ours_attn_downsampled - ours_attn_downsampled.min()) / (ours_attn_downsampled.max() - ours_attn_downsampled.min())
    ours_scores.append((((ours_attn_downsampled - human_attn_downsampled) ** 2).mean())**0.5)

print(f"RMSE of the network on the {len(os.listdir(root))} test samples: {np.mean(ours_scores):.4f} with std {np.std(ours_scores):.4f}")
