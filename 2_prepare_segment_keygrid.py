# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:56:16 2021

@author: eliphat
"""
import random
import math
import argparse
import contextlib
import torch
import torch.optim as optim
import numpy
import numpy as np
from merger.data_flower import all_h5
from merger.merger_net import Net
import merger.merger_net as merger_net

from merger.composed_chamfer import loss_all

from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset

import pdb


arg_parser = argparse.ArgumentParser(description="Training Key_Grid for the PointNet++ on the ShapeNet dataset.")
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='./Key_Grid/model/foldpant.pt',
                        help='Model checkpoint file path for saving.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=8,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-b', '--batch', type=int, default=8,
                        help='Batch size.')
arg_parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of epochs to train.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Pytorch device for predicting.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')
arg_parser.add_argument("--keynumber", type=int, help="", default=12)
arg_parser.add_argument("--chamfer", type=int, help="", default=20)
arg_parser.add_argument("--lambda_init_points", type=float, help="", default=1.0)
arg_parser.add_argument("--lambda_chamfer", type=float, help="", default=1.0)

# args for generating segmentations
arg_parser.add_argument('--dist_threshold', type=float, default=0.10, help='w.r.t. the normalized distance')
arg_parser.add_argument('--expand', type=float, default=1.2, help='the coordinates of the keypoints are expanded so they are stretched to the edges')
arg_parser.add_argument('--segment-output', type=str, default='/home/henry/robot/end2end/USEEK/data/saved_segments/pants_expand.npz')

def L2(embed):
    return 0.01 * (torch.sum(embed ** 2))




def generate_keypoints(dataloader):
    with torch.no_grad():
        keygrid_net.eval()
        for batch_id, batch_pcd in enumerate(tqdm(dataloader)):
            batch_pcd = batch_pcd.to(args.device, dtype=torch.float32)
            batch_keypoints, _= keygrid_net(batch_pcd, 'True')
            #pdb.set_trace()
            if batch_id == 0:
                keypoints = batch_keypoints.cpu().numpy()
            else:
                keypoints = np.concatenate((keypoints, batch_keypoints.cpu().numpy()), axis=0)  # [D, N, K]
    return keypoints

def generate_segments(pcds, keypoints):
    def generate_segment_mask(keypoint, pcd):
        """
        :argument
        keypoint: K x 3
        pcd: N x 3
        :return
        mask: N x K
        """
        distances = np.sqrt(np.sum(np.square(pcd.reshape(-1, 1, 3) - keypoint.reshape(1, -1, 3)), axis=-1))  # [N, K]
        # Normalized distance
        distances = (distances - np.min(distances, axis=0)) / (np.max(distances, axis=0) - np.min(distances, axis=0))
        # Fill the mask
        mask = np.where(distances < args.dist_threshold, 1., 0.)
        return mask

    segment_masks = []
    for pcd, keypoint in zip(pcds, keypoints):
        segment_mask = generate_segment_mask(keypoint[:, :] * args.expand, pcd)  # [N, K]
        segment_masks.append(segment_mask)
    return np.asarray(segment_masks)
            
if __name__ == '__main__':
    args = arg_parser.parse_args()
    batch = args.batch
    
    x = np.loadtxt("./dataset/fold/pant_seqdata.txt")# 14 is the chair dataset.
    x = x.reshape(int(x.shape[0]/args.max_points), args.max_points, 3)
    train_dataloader = DataLoader(x, batch_size=args.batch, shuffle=False)
    
    
    #net = Net(args.max_points, args.n_keypoint).cuda()
    keygrid_net = merger_net.Net(args.max_points, args.n_keypoint).to(args.device)
    keygrid_net.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device(args.device))['model_state_dict'])
    keygrid_net.eval()
   
   
    train_keypoints = generate_keypoints(train_dataloader) 
    print(f"train_keypoints shape: {train_keypoints.shape}")# (127, 8, 3)
    train_segments = generate_segments(x, train_keypoints)
    
    np.savez(args.segment_output, train_pcds=x, test_pcds=x, train_segments=train_segments, test_segments=train_segments)
    
        
        
            
