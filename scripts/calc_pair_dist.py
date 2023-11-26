import os
import numpy as np
import pickle as pkl
from os.path import join as pjoin
from scipy.spatial.distance import cdist
from magicbox.io.io import CiftiReader
from cxy_visual_dev.lib.predefine import proj_dir,\
    get_rois, mmp_map_file, Hemi2stru, mmp_name2label

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'pair_dist')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def gradient_distance(Hemi, pc_names):
    """
    Calculate gradient distance between each pair of visual cortex vertices
    PC1: absolute difference between primary gradient values of two vertices
    PC2: absolute difference between secondary gradient values of two vertices
    PCN: 以此类推

    Args:
        Hemi (str): L or R.
            L: left visual cortex
            R: right visual cortex
        pc_names (sequence):
            PC1~N
    """
    # settings
    vis_name = f'MMP-vis3-{Hemi}'
    pc_file = pjoin(
        anal_dir, 'decomposition/'
        f'HCPY-M+corrT_{vis_name}_zscore1_PCA-subj.dscalar.nii')
    out_files = pjoin(work_dir, 'gradient_distance_{pc_name}_{Hemi}.pkl')

    # prepare visual vertices' vertex numbers
    vis_rois = get_rois(vis_name)
    mmp_map = CiftiReader(mmp_map_file).get_data(Hemi2stru[Hemi], True)[0]
    vis_mask = np.zeros_like(mmp_map, bool)
    for vis_roi in vis_rois:
        vis_mask = np.logical_or(
            vis_mask, mmp_map == mmp_name2label[vis_roi])
    vis_vertices = np.where(vis_mask)[0]

    # prepare pc maps
    pc_reader = CiftiReader(pc_file)
    pc_maps = pc_reader.get_data(Hemi2stru[Hemi], True)
    vis_maps = pc_maps[:, vis_vertices].T
    map_names = pc_reader.map_names()

    # calculating
    for pc_name in pc_names:
        map_idx = map_names.index(pc_name[1:])
        vis_map = vis_maps[:, [map_idx]]
        dist_arr = cdist(vis_map, vis_map, 'euclidean')
        out_dict = {
            f'vtx_number_in_32k_fs_{Hemi}': vis_vertices,
            'mat_shape': '(n_vtx, n_vtx)',
            'mat': dist_arr}
        out_file = open(out_files.format(pc_name=pc_name, Hemi=Hemi), 'wb')
        pkl.dump(out_dict, out_file)


def geodesic_distance(Hemi):
    """
    Get geodesic distance between each pair of visual cortex vertices

    Args:
        Hemi (str): L or R.
            L: left visual cortex
            R: right visual cortex
    """
    # 直接用以下文件就行
    # vis_name = f'MMP-vis3-{Hemi}'
    # pjoin(anal_dir, f'gdist/gdist-between-all-pair-vtx_{vis_name}.pkl')
    pass


if __name__ == '__main__':
    gradient_distance('L', [f'PC{i}' for i in range(1, 6)])
    gradient_distance('R', [f'PC{i}' for i in range(1, 6)])
