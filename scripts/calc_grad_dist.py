import os
import numpy as np
import pickle as pkl
import nibabel as nib

from os.path import join as pjoin
from scipy.stats import zscore
from scipy.spatial.distance import cdist
from cxy_visual_dev.lib.predefine import proj_dir, Atlas,\
    get_rois

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'grad_dist')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def gradient_distance(Hemi):
    """
    Calculate gradient distance between each pair of visual cortex vertices
    PC1: absolute difference between primary gradient values of two vertices
    PC1-zscore: absolute difference between primary gradient values 
        (after zscore) of two vertices
    PC2: absolute difference between secondary gradient values of two vertices
    PC2-zscore: absolute difference between secondary gradient values 
        (after zscore) of two vertices
    2D-PC: euclidean distance in the 2D gradient space constructed by
        the primary and secondary gradients.
    2D-PC-zscore: euclidean distance in the 2D gradient space constructed by
        the primary and secondary gradients after zscore.

    Args:
        Hemi (str): L or R.
            L: left visual cortex
            R: right visual cortex
    """
    vis_name = f'MMP-vis3-{Hemi}'
    pc_file = pjoin(
        anal_dir,
        f'decomposition/HCPY-M+corrT_{vis_name}_zscore1_PCA-subj.dscalar.nii')
    out_file = pjoin(work_dir, f'grad_dist_{vis_name}.pkl')

    vis_mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
    vis_vertices = np.where(vis_mask)[0]
    pc_maps = nib.load(pc_file).get_fdata()[:2, vis_vertices].T
    pc_maps_z = zscore(pc_maps, 0)

    X = pc_maps[:, [0]]
    pc1_arr = cdist(X, X, 'euclidean')
    X = pc_maps[:, [1]]
    pc2_arr = cdist(X, X, 'euclidean')

    X = pc_maps_z[:, [0]]
    pc1_z_arr = cdist(X, X, 'euclidean')
    X = pc_maps_z[:, [1]]
    pc2_z_arr = cdist(X, X, 'euclidean')

    pc12_arr = cdist(pc_maps, pc_maps, 'euclidean')
    pc12_z_arr = cdist(pc_maps_z, pc_maps_z, 'euclidean')

    data = {
        'row-idx_to_32k-fs-LR-idx': vis_vertices,
        'PC1': pc1_arr, 'PC2': pc2_arr,
        'PC1-zscore': pc1_z_arr, 'PC2-zscore': pc2_z_arr,
        '2D-PC': pc12_arr, '2D-PC-zscore': pc12_z_arr}
    pkl.dump(data, open(out_file, 'wb'))


def gradient_difference(Hemi):
    """
    Calculate gradient difference between each pair of visual cortex vertices
    PC1: difference between primary gradient values of two vertices
    PC1-zscore:  difference between primary gradient values 
        (after zscore) of two vertices
    PC2: difference between secondary gradient values of two vertices
    PC2-zscore: difference between secondary gradient values 
        (after zscore) of two vertices
    以n_vtx x n_vtx的矩阵存储，其中第i行第j列的元素代表第i个顶点的梯度值减去
    第j个顶点的梯度值得到的差值。

    Args:
        Hemi (str): L or R.
            L: left visual cortex
            R: right visual cortex
    """
    vis_name = f'MMP-vis3-{Hemi}'
    pc_file = pjoin(
        anal_dir,
        f'decomposition/HCPY-M+corrT_{vis_name}_zscore1_PCA-subj.dscalar.nii')
    out_file = pjoin(work_dir, f'grad_diff_{vis_name}.pkl')

    # loading
    vis_mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
    vis_vertices = np.where(vis_mask)[0]
    pc_maps = nib.load(pc_file).get_fdata()[:2, vis_vertices]
    pc_maps_z = zscore(pc_maps, 1)

    # doing
    n_vtx = len(vis_vertices)
    arr_shape = (n_vtx, n_vtx)
    data = {
        'row-idx_to_32k-fs-LR-idx': vis_vertices,
        'PC1': np.zeros(arr_shape),
        'PC2': np.zeros(arr_shape),
        'PC1-zscore': np.zeros(arr_shape),
        'PC2-zscore': np.zeros(arr_shape)}
    for i in range(n_vtx):
        for j in range(n_vtx):
            data['PC1'][i, j] = pc_maps[0, i] - pc_maps[0, j]
            data['PC2'][i, j] = pc_maps[1, i] - pc_maps[1, j]
            data['PC1-zscore'][i, j] = pc_maps_z[0, i] - pc_maps_z[0, j]
            data['PC2-zscore'][i, j] = pc_maps_z[1, i] - pc_maps_z[1, j]

    pkl.dump(data, open(out_file, 'wb'))


if __name__ == '__main__':
    # gradient_distance(Hemi='R')  # 与prepare_plot.py中得到的数据核对过了，是一样的
    # gradient_distance(Hemi='L')
    gradient_difference(Hemi='L')
    gradient_difference(Hemi='R')
