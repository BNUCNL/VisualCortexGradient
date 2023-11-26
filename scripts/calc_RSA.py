import os
import numpy as np
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.spatial.distance import cdist
from magicbox.io.io import CiftiReader
from cxy_visual_dev.lib.predefine import proj_dir, Atlas,\
    get_rois, LR_count_32k

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'RSA')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def get_PC_RDM(Hemi):
    """
    分别计算整个视觉皮层，early，dorsal，lateral和
    ventral内部两两HCP MMP ROI之间的PC1和PC2的欧氏距离
    得到RDM。
    依据HCP MMP的22组，将HCP-MMP-visual3分成四份：
    Early: Group1+2
    Dorsal: Group3+16+17+18
    Lateral: Group5
    Ventral: Group4+13+14
    """
    pc_names = ['PC1', 'PC2']
    vis_name = f'MMP-vis3-{Hemi}'
    pc_file = pjoin(
        anal_dir, 'decomposition/'
        f'HCPY-M+corrT_{vis_name}_zscore1_PCA-subj.dscalar.nii')
    mask_names = (vis_name, 'early', 'dorsal', 'lateral', 'ventral')
    mask_name2grp = {
        'early': ['MMP-vis3-G1', 'MMP-vis3-G2'],
        'dorsal': ['MMP-vis3-G3', 'MMP-vis3-G16',
                   'MMP-vis3-G17', 'MMP-vis3-G18'],
        'lateral': ['MMP-vis3-G5'],
        'ventral': ['MMP-vis3-G4', 'MMP-vis3-G13',
                    'MMP-vis3-G14']
    }
    out_file = pjoin(work_dir, f'{vis_name}-EDLV_RDM-PC12.pkl')

    atlas = Atlas('HCP-MMP')
    vis_rois = get_rois(vis_name)
    roi2mask = {}
    for roi in vis_rois:
        roi2mask[roi] = atlas.get_mask(roi)[0]
    pc_maps = nib.load(pc_file).get_fdata()[:2]
    out_dict = {}
    for mask_name in mask_names:
        out_dict[mask_name] = {}
        if mask_name == vis_name:
            rois = vis_rois
        else:
            rois = []
            for grp in mask_name2grp[mask_name]:
                rois.extend(get_rois(grp))
            rois = [f'{Hemi}_{i}' for i in rois]
        n_roi = len(rois)
        for pc_idx, pc_name in enumerate(pc_names):
            pc_data = np.zeros((n_roi, 1))
            for roi_idx, roi in enumerate(rois):
                pc_data[roi_idx, 0] = np.mean(
                    pc_maps[pc_idx, roi2mask[roi]])
            out_dict[mask_name][pc_name] = cdist(
                pc_data, pc_data, metric='euclidean')

    pkl.dump(out_dict, open(out_file, 'wb'))


def get_WM_cope_RDM(Hemi):
    """
    分别计算整个视觉皮层，early，dorsal，lateral和
    ventral内部两两HCP MMP ROI的以WM任务四个条件激活
    构成的特征向量之间的欧氏距离得到RDM。
    依据HCP MMP的22组，将HCP-MMP-visual3分成四份：
    Early: Group1+2
    Dorsal: Group3+16+17+18
    Lateral: Group5
    Ventral: Group4+13+14
    """
    copes = ['BODY', 'FACE', 'PLACE', 'TOOL']
    cope_file = pjoin(anal_dir, 'tfMRI/tfMRI-WM-cope.dscalar.nii')
    vis_name = f'MMP-vis3-{Hemi}'
    mask_names = (vis_name, 'early', 'dorsal', 'lateral', 'ventral')
    mask_name2grp = {
        'early': ['MMP-vis3-G1', 'MMP-vis3-G2'],
        'dorsal': ['MMP-vis3-G3', 'MMP-vis3-G16',
                   'MMP-vis3-G17', 'MMP-vis3-G18'],
        'lateral': ['MMP-vis3-G5'],
        'ventral': ['MMP-vis3-G4', 'MMP-vis3-G13',
                    'MMP-vis3-G14']
    }
    out_file = pjoin(work_dir, f'{vis_name}-EDLV_RDM-WM-cope.pkl')

    # prepare roi mask
    atlas = Atlas('HCP-MMP')
    vis_rois = get_rois(vis_name)
    roi2mask = {}
    for roi in vis_rois:
        roi2mask[roi] = atlas.get_mask(roi)[0]

    # prepare cope data
    n_cope = len(copes)
    reader = CiftiReader(cope_file)
    cope_maps = reader.get_data()[:, :LR_count_32k]
    map_names = reader.map_names()
    cope_indices = [map_names.index(i) for i in copes]

    out_dict = {}
    for mask_name in mask_names:
        if mask_name == vis_name:
            rois = vis_rois
        else:
            rois = []
            for grp in mask_name2grp[mask_name]:
                rois.extend(get_rois(grp))
            rois = [f'{Hemi}_{i}' for i in rois]
        n_roi = len(rois)
        cope_data = np.zeros((n_roi, n_cope))
        for roi_idx, roi in enumerate(rois):
            for cope_i, cope_idx in enumerate(cope_indices):
                cope_data[roi_idx, cope_i] = np.mean(
                    cope_maps[cope_idx, roi2mask[roi]])
        out_dict[mask_name] = cdist(
            cope_data, cope_data, metric='euclidean')

    pkl.dump(out_dict, open(out_file, 'wb'))


if __name__ == '__main__':
    # get_PC_RDM(Hemi='R')
    get_WM_cope_RDM(Hemi='R')
