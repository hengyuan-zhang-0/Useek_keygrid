import os
import numpy as np
import h5py
import argparse
import open3d as o3d
import visualizations as visualizer


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--prediction_dir', type=str, default='/home/henry/robot/end2end/USEEK/data/prediction/pants_weight.npz')
arg_parser.add_argument('--idx', type=int, default=0)       # 3, 4, 5, 6, 8:four engines
# arg_parser.add_argument('--top', type=int, default=250)
arg_parser.add_argument('--kp_idx', type=int, default=0)
args = arg_parser.parse_args()


if __name__ == '__main__':
    pred_dataset = np.load(args.prediction_dir)

    pcd = pred_dataset['pcd']
    # gt = pred_dataset['gt'][args.idx]
    pred = pred_dataset['pred']
    
    for i in range(pred.shape[0]):

        visualizer.save_kp_and_pc_in_pcd(pcd[i],pred[i], '{}_visualizations'.format(r'./visual/pant'), save=True,name="{}".format(i))

