import os
import numpy as np
import pickle as pkl
import nibabel as nib

from os.path import join as pjoin
from magicbox.io.io import CiftiReader, GiftiReader
from magicbox.graph.triangular_mesh import get_n_ring_neighbor
from cxy_visual_dev.lib.predefine import proj_dir,\
    Hemi2stru, mmp_map_file, s1200_midthickness_L,\
    s1200_midthickness_R, mmp_name2label, mmp_label2name,\
    get_rois, Atlas

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'ROI_neighbor')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def find_ROI_neighbor():
    """
    找出HCP MMP 1.0的脑区邻接关系
    """
    Hemis = ('L', 'R')
    Hemi2hemi = {'L': 'lh', 'R': 'rh'}
    Hemi2geo = {
        'L': s1200_midthickness_L,
        'R': s1200_midthickness_R}
    out_file = pjoin(work_dir, 'HCP-MMP_ROI-neighbor.pkl')

    reader = CiftiReader(mmp_map_file)
    Hemi2mmp_map = {}
    Hemi2neighbors_list = {}
    for Hemi in Hemis:
        Hemi2mmp_map[Hemi] = \
            reader.get_data(Hemi2stru[Hemi], True)[0]
        gii = GiftiReader(Hemi2geo[Hemi])
        Hemi2neighbors_list[Hemi] = \
            get_n_ring_neighbor(gii.faces, 1)

    data = {}
    for name, lbl in mmp_name2label.items():
        Hemi = name[0]
        mmp_map = Hemi2mmp_map[Hemi]
        neighbors_list = Hemi2neighbors_list[Hemi]
        roi_vertices = np.where(mmp_map == lbl)[0]
        roi_neighbors = []
        for roi_vtx in roi_vertices:
            roi_neighbors.extend(neighbors_list[roi_vtx])
        roi_neigh_lbls = np.unique(mmp_map[roi_neighbors])
        data[name] = [mmp_label2name[i] for i in roi_neigh_lbls
                      if i not in (lbl, 0)]

    pkl.dump(data, open(out_file, 'wb'))


def get_neighbor_contrast(vis_name):
    """
    为vis_name中的每个ROI计算其与近邻ROI的梯度差异
    存在一个n_roi x n_roi的矩阵中。第i行第j列代表
    第i个ROI的梯度值减去第j个ROI的梯度值(i!=j)或是
    第i个ROI自身的标准差(i==j)。
    NAN代表此处无邻接
    """
    rois = get_rois(vis_name)
    atlas = Atlas('HCP-MMP')
    neighbor_file = pjoin(work_dir, 'HCP-MMP_ROI-neighbor.pkl')
    pc_names = ['PC1', 'PC2']
    pc_file = pjoin(
        anal_dir,
        f'decomposition/HCPY-M+corrT_{vis_name}_zscore1_PCA-subj.dscalar.nii')
    out_file = pjoin(work_dir, f'neighbor-contrast_{vis_name}.pkl')

    n_roi = len(rois)
    neighbor_dict = pkl.load(open(neighbor_file, 'rb'))
    pc_maps = nib.load(pc_file).get_fdata()[:2]
    roi2mask = {}
    for roi in rois:
        roi2mask[roi] = atlas.get_mask(roi)[0]

    data = {'roi': rois}
    for pc_idx, pc_name in enumerate(pc_names):
        data[pc_name] = np.ones((n_roi, n_roi)) * np.nan
        pc_map = pc_maps[pc_idx]
        for roi_idx1, roi1 in enumerate(rois):
            roi_data1 = pc_map[roi2mask[roi1]]
            data[pc_name][roi_idx1, roi_idx1] = np.std(roi_data1, ddof=1)
            roi_mean1 = np.mean(roi_data1)
            for nb in neighbor_dict[roi1]:
                if nb in rois:
                    roi_idx2 = rois.index(nb)
                    roi_mean2 = np.mean(pc_map[roi2mask[nb]])
                    data[pc_name][roi_idx1, roi_idx2] = \
                        roi_mean1 - roi_mean2
    pkl.dump(data, open(out_file, 'wb'))


if __name__ == '__main__':
    # find_ROI_neighbor()
    get_neighbor_contrast('MMP-vis3-R')
