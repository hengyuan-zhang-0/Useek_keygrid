# -*- coding: utf-8 -*-
import torch
import merger.merger_net as merger_net
from merger.sprin.model import SPRINSeg
import json
import tqdm
import numpy as np
import argparse
#from pytorch3d.transforms import Rotate, random_rotations
import open3d as o3d
from dgl.geometry import farthest_point_sampler
import open3d as o3d

from utils import naive_read_pcd
import pdb

arg_parser = argparse.ArgumentParser(description="Predictor for Skeleton Merger on KeypointNet dataset. Outputs a npz file with two arrays: kpcd - (N, k, 3) xyz coordinates of keypoints detected; nfact - (N, 2) normalization factor, or max and min coordinate values in a point cloud.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-a', '--annotation-json', type=str, default='./data/keypointnet/annotations/airplane.json',
                        help='Annotation JSON file path from KeypointNet dataset.')
arg_parser.add_argument('-i', '--pcd-path', type=str, default='./data/keypointnet/pcds',
                        help='Point cloud file folder path from KeypointNet dataset.')
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='./data/saved_models/pants_useek.pt',
                        help='Model checkpoint file path to load.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Pytorch device for predicting.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=8,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-b', '--batch', type=int, default=8,
                        help='Batch size.')
arg_parser.add_argument('-p', '--prediction-output', type=str, default='./data/prediction/pants_weight.npz',
                        help='Output file where prediction results are written.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')

arg_parser.add_argument('--segment', action='store_true', default=False, help='Use the segmentation method.')
arg_parser.add_argument('--weight', action='store_true', default=True, help='Use the weight method.')
arg_parser.add_argument('--nms_threshold', type=float, default=0.1)
arg_parser.add_argument('--encoder-type', type=str, default='PointNet2', choices=['PointNet2', 'DGCNN', 'EQCNN', 'SPRIN'])
arg_parser.add_argument('--do-rotation', action='store_true', default=True)
# registration methods
arg_parser.add_argument('--do-ransac_icp', action='store_true', default=False)
# ISS, http://www.open3d.org/docs/release/tutorial/geometry/iss_keypoint_detector.html?highlight=iss
arg_parser.add_argument('--iss', action='store_true', default=False)
args = arg_parser.parse_args()


# For airplane
# kp_semantic_idxes = [0, 2, 3, 5, 6]
# prediction_select_idxes = [0, 1, 2, 3, 4]
# For chair
# kp_semantic_idxes = [0, 1, 2, 3, 4, 5, 17, 18, 20]
# prediction_select_idxes = [0, 1, 3, 7, 8, 9, 2, 5, 6]
# For guitar
# kp_semantic_idxes = [0, 1, 4, 5]
# prediction_select_idxes = [1, 2, 3, 5]

# For knife
kp_semantic_idxes = [0, 1, 2]
# prediction_select_idxes = [0, 2, 3]
prediction_select_idxes = [0, 1, 2]


def semantic_check(kp_infos):
    point_idxes = []
    for kp_info in kp_infos:
        if kp_info['semantic_id'] in kp_semantic_idxes:
            point_idxes.append(kp_info['pcd_info']['point_index'])
    if len(point_idxes) != len(kp_semantic_idxes):
        return None
    else:
        return point_idxes


def segment_post_process(preds, pcds):
    all_keypoints = []
    for pred, pcd in zip(preds, pcds):      # pred: (N, K), pcd: (N, 3)
        keypoints = torch.empty(args.n_keypoint, 3)
        while not pred.isinf().all():
            position = torch.nonzero(pred == pred.max()).squeeze()
            point = pcd[position[0], :]
            keypoints[position[1], :] = point
            pred[:, position[1]] = - torch.inf

            # non maximum suppression
            dist = torch.sqrt(torch.sum(torch.square(pcd - point), dim=1))
            pred[torch.nonzero(dist < args.nms_threshold).squeeze(), :] = - torch.inf
        all_keypoints.append(keypoints)
    return torch.stack(all_keypoints)


def apply_se3(g, a):
    # g : SE(3),  4 x 4
    # a : pcd,    N x 3
    R = g[0:3, 0:3]
    p = g[0:3, 3:]
    b = np.matmul(R, np.transpose(a)) + p
    return np.transpose(b)


def RANSAC_ICP(source_pcd, target_pcd, refine=True):
    def preprocess_point_cloud(pcd, voxel_size):
        pcd_down = pcd.voxel_down_sample(voxel_size)
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def prepare_dataset(voxel_size, source_np, target_np):
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_np)
        target.points = o3d.utility.Vector3dVector(target_np)
        # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
        #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        # source.transform(trans_init)

        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
        return source, target, source_down, target_down, source_fpfh, target_fpfh

    def execute_global_registration(source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result

    def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.4
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, result_ransac.transformation)
        return result

    voxel_size = 0.05  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        voxel_size, source_pcd, target_pcd)
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    if not refine:
        return result_ransac
    result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                     voxel_size)
    return result_icp


if __name__ == '__main__':
    x = np.loadtxt("/home/henry/robot/end2end/Key-Grid-main/dataset/fold/pant_seqdata.txt")
    x = x.reshape(int(x.shape[0]/2048), 2048, 3)

    kpn_ds = x
           
    out_pred = []
    out_gt = []
    out_kpn_idx = []
    out_pcd = []
    all_gt_idxes = []

    for i in tqdm.tqdm(range(0, len(kpn_ds), args.batch), unit_scale=args.batch):
        batch_pcd = []
        for j in range(args.batch):
            if i + j >= len(kpn_ds):
                continue
            entry = kpn_ds[i + j]
            pc =  np.asarray(entry)
        
            # normalization
            pcmax = pc.max()
            pcmin = pc.min()
            pcn = (pc - pcmin) / (pcmax - pcmin)
            pcn = 2.0 * (pcn - 0.5)
            batch_pcd.append(pcn)
        
        batch_pcd = torch.from_numpy(np.array(batch_pcd)).float().to(args.device)

        if len(batch_pcd) == 0:
            continue
        # # random rotation
        # batch_pcd = torch.from_numpy(np.array(batch_pcd)).float().to(args.device)   # (B, N, 3)
        # if args.do_rotation:
        #     batch_rotations = Rotate(random_rotations(batch_pcd.shape[0], dtype=torch.float64), device=args.device)
        #     batch_pcd = batch_rotations.transform_points(batch_pcd.float())

        # # registration
        # registered_pcds = []
        # target_pcd = batch_pcd[np.random.randint(0, batch_pcd.shape[0]), :, :].cpu().numpy()
        # if args.do_ransac_icp:
        #     for pcd in batch_pcd:
        #         pcd = pcd.cpu().numpy()
        #         registered_pcds.append(apply_se3(RANSAC_ICP(pcd, target_pcd).transformation, pcd))
        #     batch_pcd = torch.from_numpy(np.asarray(registered_pcds)).float().to(args.device)

        # predict
        if args.iss:
            for pcd in batch_pcd:
                the_pcd = o3d.geometry.PointCloud()
                the_pcd.points = o3d.utility.Vector3dVector(pcd.cpu().numpy())
                raw_keypoints = np.asarray(o3d.geometry.keypoint.compute_iss_keypoints(the_pcd).points)
                idx = farthest_point_sampler(torch.from_numpy(raw_keypoints).unsqueeze(0), args.n_keypoint).squeeze().numpy()
                out_pred.append(raw_keypoints[idx])
                out_pcd.append(pcd.cpu().numpy())
        else:
            if args.segment or args.weight:
                net = SPRINSeg(args.n_keypoint).to(args.device)
            else:
                net = merger_net.Net(args.max_points, args.n_keypoint, args.encoder_type).to(args.device)
            #pdb.set_trace()
            net.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device(args.device))['model_state_dict'])
            net.eval()
            with torch.no_grad():
                if args.segment:
                    batch_pred, _ = net(batch_pcd)    # (B, N, K)
                    batch_pred = segment_post_process(batch_pred, batch_pcd)    # (B, K, 3)
                elif args.weight:
                    #pdb.set_trace()
                    batch_weight, _ = net(batch_pcd)
                    batch_weight = torch.softmax(batch_weight, dim=-2)  # (B, N, K)
                    batch_pred = torch.einsum('bnd,bnk->bkd', batch_pcd, batch_weight)    # (B, K, 3)
                else:
                    _, batch_pred, _, _, _ = net(batch_pcd)   # (B, K, 3)
                    print(batch_pred.shape)
            for pred, pcd in zip(batch_pred, batch_pcd):
                out_pred.append(pred[:, :].cpu().numpy())
                out_pcd.append(pcd.cpu().numpy())

    # # get ground truth
    # for gt_idxes, pcd in zip(all_gt_idxes, out_pcd):
    #     kps = []
    #     for idx in gt_idxes:
    #         kps.append(pcd[idx, :])
    #     out_gt.append(kps)
    #pdb.set_trace()

    np.savez(args.prediction_output, pred=np.asarray(out_pred),  pcd=np.asarray(out_pcd))
