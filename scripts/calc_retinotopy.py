import os
import numpy as np
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import pearsonr, zscore
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import proj_dir, Atlas,\
    s1200_avg_eccentricity, LR_count_32k, get_rois,\
    s1200_avg_RFsize, mmp_map_file, s1200_avg_angle

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'retinotopy')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def PC12_corr_retinotopy_roi(meas_type, roi_type):
    """
    计算PC1和PC2与retinotopic指标在区域内和区域间的相关

    Args:
        meas_type (str): retinotopic指标类型
        roi_type (str): 区域类型

    返回一个pickle文件和一个Cifti文件。前者分开存了左右脑的ROI；
    各ROI的retinotopic测量值，PC1和PC2的值；各ROI内PC1和PC2与
    retinotopic测量值的相关与显著性值。后者是把脑区各顶点赋值为
    该脑区内PC和retinotopic测量值的相关值。
    """
    Hemis = ('L', 'R')
    pc_names = ('PC1', 'PC2')
    pc_file = pjoin(
        anal_dir, 'decomposition/'
        'HCPY-M+corrT_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    meas2file = {
        'RFsize': s1200_avg_RFsize,
        'ECC': s1200_avg_eccentricity,
        'angle': s1200_avg_angle}
    out_name = f'PC12-corr-{meas_type}_roi-{roi_type}'
    out_file1 = pjoin(work_dir, f'{out_name}.pkl')
    out_file2 = pjoin(work_dir, f'{out_name}.dscalar.nii')

    n_pc = len(pc_names)
    meas_map = nib.load(
        meas2file[meas_type]).get_fdata()[0, :LR_count_32k]
    if roi_type == 'MMP-vis3':
        atlas = Atlas('HCP-MMP')
    elif roi_type == 'Wang2015':
        atlas = Atlas('Wang2015')
    else:
        raise ValueError('not supported roi_type')

    out_dict = {'pc_name': pc_names}
    out_maps = np.zeros((n_pc*2, LR_count_32k))
    map_names = [f'{i}-corr-{meas_type}' for i in pc_names]
    map_names.extend([f'abs({i})' for i in map_names])
    for Hemi in Hemis:
        mask_name = f'{roi_type}-{Hemi}'
        rois = get_rois(mask_name)
        if roi_type == 'Wang2015':
            vis_name = f'MMP-vis3-{Hemi}'
            vis_mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
            rois_tmp = []
            roi2mask = {}
            for roi in rois:
                discard_flag = False
                mask = np.logical_and(
                    vis_mask, atlas.get_mask(roi)[0])
                n_vtx = np.sum(mask)
                if n_vtx < 30:
                    print(f'#vtx in {roi} < 30')
                    discard_flag = True
                if not discard_flag:
                    rois_tmp.append(roi)
                    roi2mask[roi] = mask
            rois = rois_tmp
        n_roi = len(rois)

        pc_maps = nib.load(
            pc_file.format(Hemi=Hemi)).get_fdata()[:n_pc]
        out_dict[f'{Hemi}_ROI'] = rois
        out_dict[f'{Hemi}_{meas_type}'] = np.zeros(n_roi)
        out_dict[f'{Hemi}_PC'] = np.zeros((n_pc, n_roi))
        out_dict[f'{Hemi}_r'] = np.zeros((n_pc, n_roi))
        out_dict[f'{Hemi}_p'] = np.zeros((n_pc, n_roi))
        for roi_idx, roi in enumerate(rois):
            if roi_type == 'Wang2015':
                mask = roi2mask[roi]
            else:
                mask = atlas.get_mask(roi)[0]
            x = meas_map[mask]
            out_dict[f'{Hemi}_{meas_type}'][roi_idx] = np.mean(x)
            for pc_idx in range(n_pc):
                y = pc_maps[pc_idx, mask]
                out_dict[f'{Hemi}_PC'][pc_idx, roi_idx] = np.mean(y)
                r, p = pearsonr(x, y, alternative='two-sided')
                out_dict[f'{Hemi}_r'][pc_idx, roi_idx] = r
                out_dict[f'{Hemi}_p'][pc_idx, roi_idx] = p
                out_maps[pc_idx, mask] = r
                out_maps[pc_idx+n_pc, mask] = np.abs(r)

    pkl.dump(out_dict, open(out_file1, 'wb'))
    reader = CiftiReader(mmp_map_file)
    save2cifti(out_file2, out_maps, reader.brain_models(), map_names)


def PC12_corr_retinotopy_roi1(meas_type, roi_type):
    """
    计算PC1和PC2与retinotopic指标在区域内和区域间的相关

    Args:
        meas_type (str): retinotopic指标类型
        roi_type (str): 区域类型

    返回一个pickle文件和一个Cifti文件。前者存的是左脑和右脑
    的retinotopic测量值与PC1和PC2在区域内和区域间的相关。
    其中区域间的相关用的是区域均值。区域内的相关是先在所有区域
    内做zscore，然后拼一起做顶点级别的相关。
    后者存的是经过脑区内zscore的retinotopic, PC1和PC2 map。
    """
    Hemis = ('L', 'R')
    corr_types = ('between', 'within')
    pc_names = ('PC1', 'PC2')
    pc_file = pjoin(
        anal_dir, 'decomposition/'
        'HCPY-M+corrT_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    meas2file = {
        'RFsize': s1200_avg_RFsize,
        'ECC': s1200_avg_eccentricity}
    out_name = f'PC12-corr-{meas_type}_roi1-{roi_type}'
    out_file1 = pjoin(work_dir, f'{out_name}.pkl')
    out_file2 = pjoin(work_dir, f'{out_name}.dscalar.nii')

    n_r_type = len(corr_types)
    n_pc = len(pc_names)
    meas_map = nib.load(
        meas2file[meas_type]).get_fdata()[0, :LR_count_32k]
    if roi_type == 'MMP-vis3':
        atlas = Atlas('HCP-MMP')
    elif roi_type == 'Wang2015':
        atlas = Atlas('Wang2015')
    else:
        raise ValueError('not supported roi_type')

    out_dict = {'corr_type': corr_types, 'pc_name': pc_names,
                'arr_shape': 'n_r_type x n_pc'}
    out_maps = np.ones((n_pc + 1, LR_count_32k)) * np.nan
    map_names = pc_names + (meas_type,)
    meas_map_idx = map_names.index(meas_type)
    for Hemi in Hemis:
        mask_name = f'{roi_type}-{Hemi}'
        rois = get_rois(mask_name)
        if roi_type == 'Wang2015':
            vis_name = f'MMP-vis3-{Hemi}'
            vis_mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
            rois_tmp = []
            roi2mask = {}
            for roi in rois:
                discard_flag = False
                mask = np.logical_and(
                    vis_mask, atlas.get_mask(roi)[0])
                n_vtx = np.sum(mask)
                if n_vtx < 30:
                    print(f'#vtx in {roi} < 30')
                    discard_flag = True
                if not discard_flag:
                    rois_tmp.append(roi)
                    roi2mask[roi] = mask
            rois = rois_tmp
        n_roi = len(rois)

        pc_maps = nib.load(
            pc_file.format(Hemi=Hemi)).get_fdata()[:n_pc]
        out_dict[f'{Hemi}-r'] = np.zeros((n_r_type, n_pc))
        out_dict[f'{Hemi}-p'] = np.zeros((n_r_type, n_pc))
        meas_roi_means = np.zeros(n_roi)
        pc_roi_means = np.zeros((n_pc, n_roi))
        within_mask = np.zeros(LR_count_32k, bool)
        for roi_idx, roi in enumerate(rois):
            if roi_type == 'Wang2015':
                mask = roi2mask[roi]
            else:
                mask = atlas.get_mask(roi)[0]
            meas_roi = meas_map[mask]
            meas_roi_means[roi_idx] = np.mean(meas_roi)
            out_maps[meas_map_idx, mask] = zscore(meas_roi)
            within_mask[mask] = True
            for pc_idx, pc_name in enumerate(pc_names):
                pc_roi = pc_maps[pc_idx, mask]
                pc_roi_means[pc_idx, roi_idx] = np.mean(pc_roi)
                pc_map_idx = map_names.index(pc_name)
                out_maps[pc_map_idx, mask] = zscore(pc_roi)

        between_idx = corr_types.index('between')
        within_idx = corr_types.index('within')
        for pc_idx, pc_name in enumerate(pc_names):
            r1, p1 = pearsonr(meas_roi_means, pc_roi_means[pc_idx])
            out_dict[f'{Hemi}-r'][between_idx, pc_idx] = r1
            out_dict[f'{Hemi}-p'][between_idx, pc_idx] = p1

            pc_map_idx = map_names.index(pc_name)
            r2, p2 = pearsonr(out_maps[meas_map_idx, within_mask],
                              out_maps[pc_map_idx, within_mask])
            out_dict[f'{Hemi}-r'][within_idx, pc_idx] = r2
            out_dict[f'{Hemi}-p'][within_idx, pc_idx] = p2

    pkl.dump(out_dict, open(out_file1, 'wb'))
    reader = CiftiReader(mmp_map_file)
    save2cifti(out_file2, out_maps, reader.brain_models(), map_names)


def PC12_corr_retinotopy_roi2(meas_type, roi_type):
    """
    计算PC1和PC2与retinotopic指标在区域内和区域间的相关

    Args:
        meas_type (str): retinotopic指标类型
        roi_type (str): 区域类型

    返回一个pickle文件和一个Cifti文件。前者存的分别是左脑和右脑
    的区域内平均retinotopic测量值，PC1和PC2；以及分段内的平均。
    其中在做分段内的平均时，retinotopic测量值先在所有区域
    内做zscore，然后拼一起。
    后者存的是经过脑区内zscore的retinotopic以及
    PC1和PC2的分段map。
    """
    Hemis = ('L', 'R')
    pc_names = ('PC1', 'PC2')
    pc_file = pjoin(
        anal_dir, 'decomposition/'
        'HCPY-M+corrT_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    meas2file = {
        'RFsize': s1200_avg_RFsize,
        'ECC': s1200_avg_eccentricity}
    out_name = f'PC12-corr-{meas_type}_roi2-{roi_type}'
    out_file1 = pjoin(work_dir, f'{out_name}.pkl')
    out_file2 = pjoin(work_dir, f'{out_name}.dscalar.nii')

    n_pc = len(pc_names)
    meas_map = nib.load(
        meas2file[meas_type]).get_fdata()[0, :LR_count_32k]
    if roi_type == 'MMP-vis3':
        atlas = Atlas('HCP-MMP')
    elif roi_type == 'Wang2015':
        atlas = Atlas('Wang2015')
    else:
        raise ValueError('not supported roi_type')

    out_dict = {}
    out_maps = np.ones((1 + n_pc, LR_count_32k)) * np.nan
    map_names = [f"{meas_type} zscore"] + \
        [f'{i} segment' for i in pc_names]
    meas_out_map_idx = 0
    for Hemi in Hemis:
        out_dict[Hemi] = {}
        pc_maps = nib.load(
            pc_file.format(Hemi=Hemi)).get_fdata()[:n_pc]

        # prepare rois
        mask_name = f'{roi_type}-{Hemi}'
        rois = get_rois(mask_name)
        if roi_type == 'Wang2015':
            vis_name = f'MMP-vis3-{Hemi}'
            vis_mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
            rois_tmp = []
            roi2mask = {}
            for roi in rois:
                discard_flag = False
                mask = np.logical_and(
                    vis_mask, atlas.get_mask(roi)[0])
                n_vtx = np.sum(mask)
                if n_vtx < 30:
                    print(f'#vtx in {roi} < 30')
                    discard_flag = True
                if not discard_flag:
                    rois_tmp.append(roi)
                    roi2mask[roi] = mask
            rois = rois_tmp
        n_roi = len(rois)

        # get roi means
        out_dict[Hemi][roi_type] = {'roi': rois}
        meas_roi_means = np.zeros(n_roi)
        pc_roi_means = np.zeros((n_pc, n_roi))
        vis_mask_valid = np.zeros(LR_count_32k, bool)
        for roi_idx, roi in enumerate(rois):
            if roi_type == 'Wang2015':
                mask = roi2mask[roi]
            else:
                mask = atlas.get_mask(roi)[0]
            vis_mask_valid[mask] = True
            meas_roi = meas_map[mask]
            meas_roi_means[roi_idx] = np.mean(meas_roi)
            out_maps[meas_out_map_idx, mask] = zscore(meas_roi)
            for pc_idx, pc_name in enumerate(pc_names):
                pc_roi = pc_maps[pc_idx, mask]
                pc_roi_means[pc_idx, roi_idx] = np.mean(pc_roi)
        out_dict[Hemi][roi_type][meas_type] = meas_roi_means
        for pc_idx, pc_name in enumerate(pc_names):
            out_dict[Hemi][roi_type][pc_name] = pc_roi_means[pc_idx]

        # prepare segment indices
        n_vis_vtx = np.sum(vis_mask_valid)
        seg_indices = np.ceil(np.linspace(0, n_vis_vtx, n_roi+1))
        seg_indices = seg_indices.astype(int)
        n_seg = len(seg_indices) - 1
        assert n_seg == n_roi

        # get PC segment means
        vis_indices = np.where(vis_mask_valid)[0]
        for pc_idx, pc_name in enumerate(pc_names):
            map_name = f'{pc_name} segment'
            map_idx = map_names.index(map_name)
            out_dict[Hemi][map_name] = {}
            pc_vec = pc_maps[pc_idx][vis_indices]
            pc_sort_indices = np.argsort(pc_vec)
            meas_seg_means = np.zeros(n_seg)
            pc_seg_means = np.zeros(n_seg)
            for seg_idx, seg_idx_s in enumerate(seg_indices[:-1]):
                seg_idx_e = seg_indices[seg_idx + 1]
                pc_sort_seg_indices = pc_sort_indices[seg_idx_s:seg_idx_e]
                seg_vis_indices = vis_indices[pc_sort_seg_indices]
                meas_seg_means[seg_idx] = np.mean(
                    out_maps[meas_out_map_idx][seg_vis_indices])
                pc_seg_means[seg_idx] = np.mean(
                    pc_maps[pc_idx][seg_vis_indices])
                out_maps[map_idx][seg_vis_indices] = seg_idx
            out_dict[Hemi][map_name][meas_type] = meas_seg_means
            out_dict[Hemi][map_name][pc_name] = pc_seg_means

    # save out
    pkl.dump(out_dict, open(out_file1, 'wb'))
    reader = CiftiReader(mmp_map_file)
    save2cifti(out_file2, out_maps, reader.brain_models(), map_names)


def PC12_corr_retinotopy_roi3(meas_type, roi_type):
    """
    计算PC1和PC2与retinotopic指标在区域内和区域间的相关

    Args:
        meas_type (str): retinotopic指标类型
        roi_type (str): 区域类型

    返回一个pickle文件,存的是左脑和右脑的retinotopic测量值
    与PC1和PC2在区域内和区域间的相关。其中区域间的相关用的是
    区域均值。区域内的相关是先在所有区域内做zscore，然后
    拼一起做顶点级别的相关（只针对retinotopic这样做）
    """
    Hemis = ('L', 'R')
    corr_types = ('between', 'within')
    pc_names = ('PC1', 'PC2')
    pc_file = pjoin(
        anal_dir, 'decomposition/'
        'HCPY-M+corrT_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    meas2file = {
        'RFsize': s1200_avg_RFsize,
        'ECC': s1200_avg_eccentricity}
    out_name = f'PC12-corr-{meas_type}_roi3-{roi_type}'
    out_file = pjoin(work_dir, f'{out_name}.pkl')

    n_r_type = len(corr_types)
    n_pc = len(pc_names)
    meas_map = nib.load(
        meas2file[meas_type]).get_fdata()[0, :LR_count_32k]
    if roi_type == 'MMP-vis3':
        atlas = Atlas('HCP-MMP')
    elif roi_type == 'Wang2015':
        atlas = Atlas('Wang2015')
    else:
        raise ValueError('not supported roi_type')

    out_dict = {'corr_type': corr_types, 'pc_name': pc_names,
                'arr_shape': 'n_r_type x n_pc'}
    meas_zscore_map = np.zeros(LR_count_32k)
    for Hemi in Hemis:
        mask_name = f'{roi_type}-{Hemi}'
        rois = get_rois(mask_name)
        if roi_type == 'Wang2015':
            vis_name = f'MMP-vis3-{Hemi}'
            vis_mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
            rois_tmp = []
            roi2mask = {}
            for roi in rois:
                discard_flag = False
                mask = np.logical_and(
                    vis_mask, atlas.get_mask(roi)[0])
                n_vtx = np.sum(mask)
                if n_vtx < 30:
                    print(f'#vtx in {roi} < 30')
                    discard_flag = True
                if not discard_flag:
                    rois_tmp.append(roi)
                    roi2mask[roi] = mask
            rois = rois_tmp
        n_roi = len(rois)

        pc_maps = nib.load(
            pc_file.format(Hemi=Hemi)).get_fdata()[:n_pc]
        out_dict[f'{Hemi}-r'] = np.zeros((n_r_type, n_pc))
        out_dict[f'{Hemi}-p'] = np.zeros((n_r_type, n_pc))
        meas_roi_means = np.zeros(n_roi)
        pc_roi_means = np.zeros((n_pc, n_roi))
        within_mask = np.zeros(LR_count_32k, bool)
        for roi_idx, roi in enumerate(rois):
            if roi_type == 'Wang2015':
                mask = roi2mask[roi]
            else:
                mask = atlas.get_mask(roi)[0]
            meas_roi = meas_map[mask]
            meas_roi_means[roi_idx] = np.mean(meas_roi)
            meas_zscore_map[mask] = zscore(meas_roi)
            within_mask[mask] = True
            for pc_idx, pc_name in enumerate(pc_names):
                pc_roi = pc_maps[pc_idx, mask]
                pc_roi_means[pc_idx, roi_idx] = np.mean(pc_roi)

        between_idx = corr_types.index('between')
        within_idx = corr_types.index('within')
        for pc_idx, pc_name in enumerate(pc_names):
            r1, p1 = pearsonr(meas_roi_means, pc_roi_means[pc_idx])
            out_dict[f'{Hemi}-r'][between_idx, pc_idx] = r1
            out_dict[f'{Hemi}-p'][between_idx, pc_idx] = p1

            r2, p2 = pearsonr(meas_zscore_map[within_mask],
                              pc_maps[pc_idx][within_mask])
            out_dict[f'{Hemi}-r'][within_idx, pc_idx] = r2
            out_dict[f'{Hemi}-p'][within_idx, pc_idx] = p2

    pkl.dump(out_dict, open(out_file, 'wb'))


if __name__ == '__main__':
    # PC12_corr_retinotopy_roi(meas_type='RFsize', roi_type='MMP-vis3')
    # PC12_corr_retinotopy_roi(meas_type='RFsize', roi_type='Wang2015')
    # PC12_corr_retinotopy_roi(meas_type='ECC', roi_type='Wang2015')
    # PC12_corr_retinotopy_roi(meas_type='angle', roi_type='Wang2015')
    # PC12_corr_retinotopy_roi1(meas_type='RFsize', roi_type='Wang2015')
    # PC12_corr_retinotopy_roi2(meas_type='RFsize', roi_type='Wang2015')
    PC12_corr_retinotopy_roi3(meas_type='RFsize', roi_type='Wang2015')
