import os
import time
import math
import numpy as np
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold,\
    KFold
from sklearn.pipeline import Pipeline
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import LR_count_32k, proj_dir, Atlas,\
    get_rois, mmp_map_file, wang2015_file, mmp_name2label

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'predict')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def get_rois_by_mask_name(mask_name):
    if mask_name == 'MMP-vis3-R':
        rois = get_rois(mask_name)
    elif mask_name == 'MMP-vis3-L':
        rois = get_rois(mask_name)

    elif mask_name == 'MMP-vis3-R_ex(V1~4)':
        rois_vis = get_rois('MMP-vis3-R')
        rois_ex = ['R_V1', 'R_V2', 'R_V3', 'R_V4']
        rois = [i for i in rois_vis if i not in rois_ex]
    elif mask_name == 'R_V1~4+V3A':
        rois = ['R_V1', 'R_V2', 'R_V3', 'R_V4', 'R_V3A']
    elif mask_name == 'MMP-vis3-R_ex(V1~4+V3A)':
        rois_vis = get_rois('MMP-vis3-R')
        rois_ex = ['R_V1', 'R_V2', 'R_V3', 'R_V4', 'R_V3A']
        rois = [i for i in rois_vis if i not in rois_ex]

    elif mask_name == 'MMP-vis3-R-early':
        rois = get_rois('MMP-vis3-G1') + get_rois('MMP-vis3-G2')
        rois = [f'R_{roi}' for roi in rois]
    elif mask_name == 'MMP-vis3-L-early':
        rois = get_rois('MMP-vis3-G1') + get_rois('MMP-vis3-G2')
        rois = [f'L_{roi}' for roi in rois]

    elif mask_name == 'MMP-vis3-R-dorsal':
        rois = get_rois('MMP-vis3-G3') + get_rois('MMP-vis3-G16') +\
            get_rois('MMP-vis3-G17') + get_rois('MMP-vis3-G18')
        rois = [f'R_{roi}' for roi in rois]
    elif mask_name == 'MMP-vis3-L-dorsal':
        rois = get_rois('MMP-vis3-G3') + get_rois('MMP-vis3-G16') +\
            get_rois('MMP-vis3-G17') + get_rois('MMP-vis3-G18')
        rois = [f'L_{roi}' for roi in rois]

    elif mask_name == 'MMP-vis3-R-lateral':
        rois = get_rois('MMP-vis3-G5')
        rois = [f'R_{roi}' for roi in rois]
    elif mask_name == 'MMP-vis3-L-lateral':
        rois = get_rois('MMP-vis3-G5')
        rois = [f'L_{roi}' for roi in rois]

    elif mask_name == 'MMP-vis3-R-ventral':
        rois = get_rois('MMP-vis3-G4') + get_rois('MMP-vis3-G13') +\
            get_rois('MMP-vis3-G14')
        rois = [f'R_{roi}' for roi in rois]
    elif mask_name == 'MMP-vis3-L-ventral':
        rois = get_rois('MMP-vis3-G4') + get_rois('MMP-vis3-G13') +\
            get_rois('MMP-vis3-G14')
        rois = [f'L_{roi}' for roi in rois]

    elif mask_name == 'MMP-vis3-R-forefront':
        rois = ['R_TF', 'R_PeEc']
    elif mask_name == 'MMP-vis3-R-ventral_ex(forefront)':
        rois = get_rois('MMP-vis3-G4') + get_rois('MMP-vis3-G13') +\
            get_rois('MMP-vis3-G14')
        rois_ex = ['TF', 'PeEc']
        rois = [f'R_{roi}' for roi in rois if roi not in rois_ex]
    else:
        raise ValueError("not supported mask name")

    return rois


def PC12_predict_ROI1():
    """
    用HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj的PC1和PC2
    预测HCP MMP1.0的脑区。将脑区内顶点以3:1的比例拆分成train和test set
    基于train set用logistic回归和linear SVC，基于各自设定的参数范围做grid search
    找到最优组合后，基于整个train set拟合模型，然后在test set上得到分数。
    数据拆分策略使用StratifiedKFold
    数据在拆分后要经过StandardScaler的处理
    """
    vis_name = 'MMP-vis3-R'
    atlas = Atlas('HCP-MMP')
    feat_file = pjoin(
        anal_dir,
        f'decomposition/HCPY-M+T_{vis_name}_zscore1_PCA-subj.dscalar.nii')

    # prepare X, y
    mask = atlas.get_mask(get_rois(vis_name))[0]
    X = nib.load(feat_file).get_fdata()[:2, mask].T
    y = atlas.maps[0, mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=7, stratify=y)

    # prepare pipeline and GridSearch
    pipe = Pipeline([('preprocesser', StandardScaler()),
                     ('classifier', LogisticRegression())])
    param_grid = [
        {'classifier': [LogisticRegression(penalty='none')],
         'classifier__solver': ['saga', 'sag', 'lbfgs', 'newton-cg']},
        {'classifier': [LogisticRegression(penalty='l1')],
         'classifier__solver': ['saga', 'liblinear'],
         'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
        {'classifier': [LogisticRegression(penalty='l2')],
         'classifier__solver': ['sag', 'saga', 'liblinear', 'lbfgs', 'newton-cg'],
         'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
        {'classifier': [SVC(kernel='linear')],
         'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy')

    # computation
    grid.fit(X_train, y_train)

    # output
    print('best estimator:\n', grid.best_estimator_)
    print('\nbest cross-validation score:', grid.best_score_)
    print('\nTest-set score:', grid.score(X_test, y_test))


def PC12_predict_ROI2():
    """
    1. 以StratifiedKFold划分Train和Test（n_split=10），对于每次split，
    用GridSearchCV在Train上搜索超参数（划分Train和Validation的方法
    也是StratifiedKFold (n_split=10)），并用最优超参数在整个Train上训练模型，
    然后得到在Test上的预测。（n_split取大一点可以让训练集大一点，提高test上的准确率。
    内部的CV用的是总体准确率来选超参数，所以不用担心有些类的validation集太小。
    而外部是汇总所有test的预测值之后再计算总体和类别准确率，所以也不怕有些类的test集太小）
    2. 所有split的基于Test的预测值合在一起就可以得到所有顶点的预测值（可以说是相当严格了）。
    基于这些预测值可以计算总体准确率和类别准确率；
    3. 然后展示这个预测结果的脑图，对比MMP的ground truth脑图，算个相关。
    """
    n_split = 10
    vis_name = 'MMP-vis3-R'
    mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
    feat_file = pjoin(
        anal_dir,
        f'decomposition/HCPY-M+T_{vis_name}_zscore1_PCA-subj.dscalar.nii')
    reader = CiftiReader(mmp_map_file)
    out_pkl = pjoin(work_dir, f'PC12_predict_ROI2_{vis_name}.pkl')
    out_cii = pjoin(work_dir, f'PC12_predict_ROI2_{vis_name}.dlabel.nii')

    # prepare X, y
    X = nib.load(feat_file).get_fdata()[:2, mask].T
    y = reader.get_data()[0, mask]

    # prepare parameters for GridSearch
    Cs = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    param_grid = [
        {'classifier': [LogisticRegression(penalty='none')],
         'classifier__solver': ['saga', 'sag', 'lbfgs', 'newton-cg']},
        {'classifier': [LogisticRegression(penalty='l1')],
         'classifier__solver': ['saga', 'liblinear'],
         'classifier__C': Cs},
        {'classifier': [LogisticRegression(penalty='l2')],
         'classifier__solver': ['sag', 'saga', 'liblinear', 'lbfgs', 'newton-cg'],
         'classifier__C': Cs},
        # 这样指定linearSVC好像只能是L2正则，以后可以直接用LinearSVC这个接口，是可以调正则方式的
        {'classifier': [SVC(kernel='linear')],
         'classifier__C': Cs}
    ]

    # loop n_split train-test splits
    out_dict = {'model': [], 'train_score': [], 'test_score': []}
    out_map = np.zeros(reader.full_data.shape, np.uint16)
    y_pred = np.zeros_like(y)
    skf = StratifiedKFold(n_split, shuffle=True, random_state=7)
    split_idx = 1
    for train_indices, test_indices in skf.split(X, y):
        time1 = time.time()
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        pipe = Pipeline([('preprocesser', StandardScaler()),
                         ('classifier', LogisticRegression())])
        grid = GridSearchCV(pipe, param_grid, cv=n_split, scoring='accuracy')
        grid.fit(X_train, y_train)
        y_pred[test_indices] = grid.predict(X_test)

        out_dict['model'].append(grid)
        out_dict['train_score'].append(grid.score(X_train, y_train))
        out_dict['test_score'].append(grid.score(X_test, y_test))
        print(f'Finished {split_idx}/{n_split}, '
              f'cost {time.time() - time1} seconds.')
        split_idx += 1
    out_map[0, mask] = y_pred

    # output
    pkl.dump(out_dict, open(out_pkl, 'wb'))
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab_old = reader.label_tables()[0]
    for k in np.unique(out_map):
        lbl_tab[k] = lbl_tab_old[k]
    save2cifti(out_cii, out_map, reader.brain_models(),
               label_tables=[lbl_tab])


def PC12_predict_ROI3():
    """
    1. 以StratifiedKFold划分Train和Test（n_split=5），对于每次split，
    用GridSearchCV在Train上搜索超参数（划分Train和Validation的方法
    也是StratifiedKFold (n_split=5)），并用最优超参数在整个Train上训练模型，
    然后得到在Test上的预测。（n_split取大一点可以让训练集大一点，提高test上的准确率。
    内部的CV用的是总体准确率来选超参数，所以不用担心有些类的validation集太小。
    而外部是汇总所有test的预测值之后再计算总体和类别准确率，所以也不怕有些类的test集太小）
    由之前的结果发现，在训练过程中，小区由于样本少，对最终正确率影响较小，因此训练出的模型
    对有些小区的预测正确率很低。于是我决定在每次split的训练集中为每个区整体复制样本点，直到其数量
    不超过最大的区的数量，这样在优化目标函数的时候，小区的重要性也在，但是信息量应该是不变的。
    不管在训练集上怎么操作，只要测试集的精度高，并且类别准确率合理，那模型就是成功的！
    2. 所有split的基于Test的预测值合在一起就可以得到所有顶点的预测值（可以说是相当严格了）。
    基于这些预测值可以计算总体准确率和类别准确率；
    3. 然后展示这个预测结果的脑图，对比MMP的ground truth脑图，算个相关。
    """
    n_split = 5
    vis_name = 'MMP-vis3-R'
    mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
    feat_file = pjoin(
        anal_dir,
        f'decomposition/HCPY-M+T_{vis_name}_zscore1_PCA-subj.dscalar.nii')
    reader = CiftiReader(mmp_map_file)
    out_pkl = pjoin(work_dir, f'PC12_predict_ROI3_{vis_name}.pkl')
    out_cii = pjoin(work_dir, f'PC12_predict_ROI3_{vis_name}.dlabel.nii')

    # prepare X, y
    X = nib.load(feat_file).get_fdata()[:2, mask].T
    y = reader.get_data()[0, mask]

    # prepare parameters for GridSearch
    Cs = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    param_grid = [
        {'classifier': [LogisticRegression(penalty='none', max_iter=1000)],
         'classifier__solver': ['saga', 'lbfgs']},
        {'classifier': [LogisticRegression(penalty='l2', max_iter=1000)],
         'classifier__solver': ['saga', 'lbfgs'],
         'classifier__C': Cs},
    ]

    # loop n_split train-test splits
    out_dict = {'model': [], 'train_score': [], 'test_score': [], 'train_score_new': []}
    out_map = np.zeros(reader.full_data.shape, np.uint16)
    y_pred = np.zeros_like(y)
    skf = StratifiedKFold(n_split, shuffle=True, random_state=7)
    split_idx = 1
    for train_indices, test_indices in skf.split(X, y):
        time1 = time.time()
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        labels = np.unique(y_train)
        label_sizes = [np.sum(y_train == lbl) for lbl in labels]
        lbl_sz_max = np.max(label_sizes)
        X_train_new = np.zeros((0, X_train.shape[1]), np.float64)
        y_train_new = np.zeros(0, np.float64)
        label_sizes_new = []
        for lbl, lbl_sz in zip(labels, label_sizes):
            idx_vec = y_train == lbl
            ratio = math.floor(lbl_sz_max / lbl_sz)
            X_train_new = np.r_[X_train_new, np.tile(X_train[idx_vec], (ratio, 1))]
            y_train_new = np.r_[y_train_new, np.tile(y_train[idx_vec], (ratio,))]
            label_sizes_new.append(lbl_sz * ratio)
        print('label size new (max):', np.max(label_sizes_new))
        print('label size new (min):', np.min(label_sizes_new))
        print('X_train_new.shape:', X_train_new.shape)
        print('y_train_new.shape:', y_train_new.shape)

        pipe = Pipeline([('preprocesser', StandardScaler()),
                         ('classifier', LogisticRegression())])
        cv = StratifiedKFold(n_split, shuffle=True, random_state=7)
        grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy')
        grid.fit(X_train_new, y_train_new)
        y_pred[test_indices] = grid.predict(X_test)

        out_dict['model'].append(grid)
        out_dict['train_score'].append(grid.score(X_train, y_train))
        out_dict['train_score_new'].append(grid.score(X_train_new, y_train_new))
        out_dict['test_score'].append(grid.score(X_test, y_test))
        print(f'Finished {split_idx}/{n_split}, '
              f'cost {time.time() - time1} seconds.')
        split_idx += 1
    out_map[0, mask] = y_pred

    # output
    pkl.dump(out_dict, open(out_pkl, 'wb'))
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab_old = reader.label_tables()[0]
    for k in np.unique(out_map):
        lbl_tab[k] = lbl_tab_old[k]
    save2cifti(out_cii, out_map, reader.brain_models(),
               label_tables=[lbl_tab])


def PC_predict_ROI4(pc_file, pc_nums, mask_name, n_split=5):
    """
    改造自PC12_predict_ROI2，使其可以指定用哪些PC预测哪些ROI
    1. 以StratifiedKFold划分Train和Test（n_split），对于每次split，
    用GridSearchCV在Train上搜索超参数（划分Train和Validation的方法
    也是StratifiedKFold (n_split)），并用最优超参数在整个Train上训练模型，
    然后得到在Test上的预测。
    2. 所有split的基于Test的预测值合在一起就可以得到所有顶点的预测值（可以说是相当严格了）。
    基于这些预测值可以计算总体准确率和类别准确率；
    3. 然后展示这个预测结果的脑图，对比MMP的ground truth脑图，算个相关。

    Args:
        pc_file (str):
        pc_nums (integers): 主成分的编号（从1开始）
            用于指定使用哪些PC
        mask_name (str): 用于指定预测哪些ROI
        n_split (int, optional): Defaults to 5.
    """
    # prepare
    pc_indices = [i - 1 for i in pc_nums]
    rois = get_rois_by_mask_name(mask_name)
    mask = Atlas('HCP-MMP').get_mask(rois)[0]
    reader = CiftiReader(mmp_map_file)
    pc_nums_str = '+'.join(map(str, pc_nums))
    out_pkl = pjoin(work_dir, f'PC_predict_ROI4_PC{pc_nums_str}_mask-{mask_name}.pkl')
    out_cii = pjoin(work_dir, f'PC_predict_ROI4_PC{pc_nums_str}_mask-{mask_name}.dlabel.nii')

    # prepare X, y
    X = nib.load(pc_file).get_fdata()[pc_indices][:, mask].T
    y = reader.get_data()[0, mask]

    # prepare parameters for GridSearch
    Cs = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    param_grid = [
        {'classifier': [LogisticRegression(penalty='none', max_iter=10000)],
         'classifier__solver': ['saga', 'lbfgs']},
        {'classifier': [LogisticRegression(penalty='l2', max_iter=10000)],
         'classifier__solver': ['saga', 'lbfgs'],
         'classifier__C': Cs},
    ]

    # loop n_split train-test splits
    out_dict = {'model': [], 'train_score': [], 'test_score': []}
    out_map = np.zeros(reader.full_data.shape, np.uint16)
    y_pred = np.zeros_like(y)
    skf = StratifiedKFold(n_split, shuffle=True, random_state=7)
    split_idx = 1
    for train_indices, test_indices in skf.split(X, y):
        time1 = time.time()
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        pipe = Pipeline([('preprocesser', StandardScaler()),
                         ('classifier', LogisticRegression())])
        grid = GridSearchCV(pipe, param_grid, cv=n_split, scoring='accuracy')
        grid.fit(X_train, y_train)
        y_pred[test_indices] = grid.predict(X_test)

        out_dict['model'].append(grid)
        out_dict['train_score'].append(grid.score(X_train, y_train))
        out_dict['test_score'].append(grid.score(X_test, y_test))
        print(f'Finished {split_idx}/{n_split}, '
              f'cost {time.time() - time1} seconds.')
        split_idx += 1
    out_map[0, mask] = y_pred
    out_dict['y_true'] = y
    out_dict['y_pred'] = y_pred
    out_dict['roi_name'] = rois

    # output
    pkl.dump(out_dict, open(out_pkl, 'wb'))
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab_old = reader.label_tables()[0]
    for k in np.unique(out_map):
        lbl_tab[k] = lbl_tab_old[k]
    save2cifti(out_cii, out_map, reader.brain_models(),
               label_tables=[lbl_tab])


def PC_predict_ROI5(pc_file, pc_nums, mask_name, n_split=5):
    """
    改造自PC12_predict_ROI3，使其可以指定用哪些PC预测哪些ROI
    1. 以StratifiedKFold划分Train和Test（n_split），对于每次split，
    用GridSearchCV在Train上搜索超参数（划分Train和Validation的方法
    也是StratifiedKFold (n_split)），并用最优超参数在整个Train上训练模型，
    然后得到在Test上的预测。（在每次split的训练集中为每个区整体复制样本点，直到其数量
    不超过最大的区的数量，这样在优化目标函数的时候，小区的重要性也在，但是信息量应该是不变的。
    不管在训练集上怎么操作，只要测试集的精度高，并且类别准确率合理，那模型就是成功的！）
    2. 所有split的基于Test的预测值合在一起就可以得到所有顶点的预测值。
    基于这些预测值可以计算总体准确率和类别准确率；
    3. 然后展示这个预测结果的脑图，对比MMP的ground truth脑图，算个相关。

    Args:
        pc_file (str):
        pc_nums (integers): 主成分的编号（从1开始）
            用于指定使用哪些PC
        mask_name (str): 用于指定预测哪些ROI
        n_split (int, optional): Defaults to 5.
    """
    pc_indices = [i - 1 for i in pc_nums]
    rois = get_rois_by_mask_name(mask_name)
    mask = Atlas('HCP-MMP').get_mask(rois)[0]
    reader = CiftiReader(mmp_map_file)
    pc_nums_str = '+'.join(map(str, pc_nums))
    out_pkl = pjoin(work_dir, f'PC_predict_ROI5_PC{pc_nums_str}_mask-{mask_name}.pkl')
    out_cii = pjoin(work_dir, f'PC_predict_ROI5_PC{pc_nums_str}_mask-{mask_name}.dlabel.nii')

    # prepare X, y
    X = nib.load(pc_file).get_fdata()[pc_indices][:, mask].T
    y = reader.get_data()[0, mask]

    # prepare parameters for GridSearch
    Cs = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    param_grid = [
        {'classifier': [LogisticRegression(penalty='none', max_iter=10000)],
         'classifier__solver': ['saga', 'lbfgs']},
        {'classifier': [LogisticRegression(penalty='l2', max_iter=10000)],
         'classifier__solver': ['saga', 'lbfgs'],
         'classifier__C': Cs},
    ]

    # loop n_split train-test splits
    out_dict = {'model': [], 'train_score': [], 'test_score': [], 'train_score_new': []}
    out_map = np.zeros(reader.full_data.shape, np.uint16)
    y_pred = np.zeros_like(y)
    skf = StratifiedKFold(n_split, shuffle=True, random_state=7)
    split_idx = 1
    for train_indices, test_indices in skf.split(X, y):
        time1 = time.time()
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        labels = np.unique(y_train)
        label_sizes = [np.sum(y_train == lbl) for lbl in labels]
        lbl_sz_max = np.max(label_sizes)
        X_train_new = np.zeros((0, X_train.shape[1]), np.float64)
        y_train_new = np.zeros(0, np.float64)
        label_sizes_new = []
        for lbl, lbl_sz in zip(labels, label_sizes):
            idx_vec = y_train == lbl
            ratio = math.floor(lbl_sz_max / lbl_sz)
            X_train_new = np.r_[X_train_new, np.tile(X_train[idx_vec], (ratio, 1))]
            y_train_new = np.r_[y_train_new, np.tile(y_train[idx_vec], (ratio,))]
            label_sizes_new.append(lbl_sz * ratio)
        print('label size new (max):', np.max(label_sizes_new))
        print('label size new (min):', np.min(label_sizes_new))
        print('X_train_new.shape:', X_train_new.shape)
        print('y_train_new.shape:', y_train_new.shape)

        pipe = Pipeline([('preprocesser', StandardScaler()),
                         ('classifier', LogisticRegression())])
        cv = StratifiedKFold(n_split, shuffle=True, random_state=7)
        grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy')
        grid.fit(X_train_new, y_train_new)
        y_pred[test_indices] = grid.predict(X_test)

        out_dict['model'].append(grid)
        out_dict['train_score'].append(grid.score(X_train, y_train))
        out_dict['train_score_new'].append(grid.score(X_train_new, y_train_new))
        out_dict['test_score'].append(grid.score(X_test, y_test))
        print(f'Finished {split_idx}/{n_split}, '
              f'cost {time.time() - time1} seconds.')
        split_idx += 1
    out_map[0, mask] = y_pred
    out_dict['y_true'] = y
    out_dict['y_pred'] = y_pred
    out_dict['roi_name'] = rois

    # output
    pkl.dump(out_dict, open(out_pkl, 'wb'))
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab_old = reader.label_tables()[0]
    for k in np.unique(out_map):
        lbl_tab[k] = lbl_tab_old[k]
    save2cifti(out_cii, out_map, reader.brain_models(),
               label_tables=[lbl_tab])


def PC_predict_ROI6(pc_file, trg, pc_nums, mask_name, n_split=5):
    """
    改造自PC12_predict_ROI5，使其可以指定用哪个target ROI map
    1. 以StratifiedKFold划分Train和Test（n_split），对于每次split，
    用GridSearchCV在Train上搜索超参数（划分Train和Validation的方法
    也是StratifiedKFold (n_split)），并用最优超参数在整个Train上训练模型，
    然后得到在Test上的预测。（在每次split的训练集中为每个区整体复制样本点，直到其数量
    不超过最大的区的数量，这样在优化目标函数的时候，小区的重要性也在，但是信息量应该是不变的。
    不管在训练集上怎么操作，只要测试集的精度高，并且类别准确率合理，那模型就是成功的！）
    2. 所有split的基于Test的预测值合在一起就可以得到所有顶点的预测值。
    基于这些预测值可以计算总体准确率和类别准确率；
    3. 然后展示这个预测结果的脑图，对比MMP的ground truth脑图，算个相关。

    Args:
        pc_file (str):
        trg (str): MPM, animate, count
        pc_nums (integers): 主成分的编号（从1开始）
            用于指定使用哪些PC
        mask_name (str): 用于指定预测哪些ROI
        n_split (int, optional): Defaults to 5.
    """
    pc_indices = [i - 1 for i in pc_nums]
    mask_rois = get_rois_by_mask_name(mask_name)
    mask = Atlas('HCP-MMP').get_mask(mask_rois)[0]
    roi_file = pjoin(anal_dir, f'tfMRI/HCPY-category_prob-map_thr2.3_summary-{trg}-thr0.2.dlabel.nii')
    reader = CiftiReader(roi_file)
    roi_map = reader.get_data()[0, :LR_count_32k]
    mask = np.logical_and(mask, roi_map != 0)
    pc_nums_str = '+'.join(map(str, pc_nums))
    out_pkl = pjoin(work_dir, f'PC_predict_ROI6_{trg}_PC{pc_nums_str}_mask-{mask_name}.pkl')
    out_cii = pjoin(work_dir, f'PC_predict_ROI6_{trg}_PC{pc_nums_str}_mask-{mask_name}.dlabel.nii')

    # prepare X, y
    X = nib.load(pc_file).get_fdata()[pc_indices][:, mask].T
    y = roi_map[mask]

    # prepare parameters for GridSearch
    Cs = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    param_grid = [
        {'classifier': [LogisticRegression(penalty='none', max_iter=1000)],
         'classifier__solver': ['saga', 'lbfgs']},
        {'classifier': [LogisticRegression(penalty='l2', max_iter=1000)],
         'classifier__solver': ['saga', 'lbfgs'],
         'classifier__C': Cs},
    ]

    # loop n_split train-test splits
    out_dict = {'model': [], 'train_score': [], 'test_score': [], 'train_score_new': []}
    out_map = np.zeros((1, LR_count_32k), np.uint16)
    y_pred = np.zeros_like(y)
    skf = StratifiedKFold(n_split, shuffle=True, random_state=7)
    split_idx = 1
    for train_indices, test_indices in skf.split(X, y):
        time1 = time.time()
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        labels = np.unique(y_train)
        label_sizes = [np.sum(y_train == lbl) for lbl in labels]
        lbl_sz_max = np.max(label_sizes)
        X_train_new = np.zeros((0, X_train.shape[1]), np.float64)
        y_train_new = np.zeros(0, np.float64)
        label_sizes_new = []
        for lbl, lbl_sz in zip(labels, label_sizes):
            idx_vec = y_train == lbl
            ratio = math.floor(lbl_sz_max / lbl_sz)
            X_train_new = np.r_[X_train_new, np.tile(X_train[idx_vec], (ratio, 1))]
            y_train_new = np.r_[y_train_new, np.tile(y_train[idx_vec], (ratio,))]
            label_sizes_new.append(lbl_sz * ratio)
        print('label size new (max):', np.max(label_sizes_new))
        print('label size new (min):', np.min(label_sizes_new))
        print('X_train_new.shape:', X_train_new.shape)
        print('y_train_new.shape:', y_train_new.shape)

        pipe = Pipeline([('preprocesser', StandardScaler()),
                         ('classifier', LogisticRegression())])
        cv = StratifiedKFold(n_split, shuffle=True, random_state=7)
        grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy')
        grid.fit(X_train_new, y_train_new)
        y_pred[test_indices] = grid.predict(X_test)

        out_dict['model'].append(grid)
        out_dict['train_score'].append(grid.score(X_train, y_train))
        out_dict['train_score_new'].append(grid.score(X_train_new, y_train_new))
        out_dict['test_score'].append(grid.score(X_test, y_test))
        print(f'Finished {split_idx}/{n_split}, '
              f'cost {time.time() - time1} seconds.')
        split_idx += 1
    out_map[0, mask] = y_pred
    out_dict['y_true'] = y
    out_dict['y_pred'] = y_pred
    keys = np.unique(y)
    lbl_tab_old = reader.label_tables()[0]
    out_dict['roi_key'] = keys
    out_dict['roi_name'] = [lbl_tab_old[k].label for k in keys]

    # output
    pkl.dump(out_dict, open(out_pkl, 'wb'))
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    for k in np.unique(out_map):
        lbl_tab[k] = lbl_tab_old[k]
    save2cifti(out_cii, out_map, CiftiReader(mmp_map_file).brain_models(),
               label_tables=[lbl_tab])


def prepare_visual_ROIs():
    """
    转存MMP-vis3中的HCP MMP脑区
    当脑区顶点数量超过300时，随机保留300
    """
    reader = CiftiReader(mmp_map_file)
    mmp_map = reader.get_data()
    rois = get_rois('MMP-vis3-L') + get_rois('MMP-vis3-R')
    out_file = pjoin(work_dir, 'HCP-MMP1_visual-cortex3_size-300.dlabel.nii')

    out_map = np.zeros_like(mmp_map, dtype=int)
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab_old = reader.label_tables()[0]
    lbl_tab[0] = lbl_tab_old[0]
    for roi in rois:
        k = mmp_name2label[roi]
        lbl_tab[k] = lbl_tab_old[k]
        roi_mask = mmp_map == k
        roi_mask_indices = np.where(roi_mask[0])[0]
        if len(roi_mask_indices) > 300:
            # 超过300个顶点的脑区随机留用300个顶点
            roi_mask_indices = np.random.choice(
                roi_mask_indices, 300, False)
        out_map[0, roi_mask_indices] = k

    save2cifti(out_file, out_map, reader.brain_models(),
               label_tables=[lbl_tab])


def PC_predict_ROI7(pc_file, pc_nums, trg, n_split=5):
    """
    用指定的PC (pc_nums)预测目标ROIs (trg)
    1. 以StratifiedKFold划分Train和Test（n_split），对于每次split，
    用GridSearchCV在Train上搜索超参数（划分Train和Validation的方法
    也是StratifiedKFold (n_split)），并用最优超参数在整个Train上训练模型，
    然后得到在Test上的预测。
    2. 所有split的基于Test的预测值合在一起就可以得到所有顶点的预测值。
    基于这些预测值可以计算总体准确率和类别准确率；
    3. 然后展示这个预测结果的脑图，对比MMP的ground truth脑图，算个相关。

    Args:
        pc_file (str):
        pc_nums (integers): 主成分的编号（从1开始）
            用于指定使用哪些PC
        trg (str): 用于指定预测哪些ROI
        n_split (int, optional): Defaults to 5.
    """
    # settings
    pc_indices = [i - 1 for i in pc_nums]
    pc_nums_str = '+'.join(map(str, pc_nums))
    out_pkl = pjoin(work_dir, f'PC_predict_ROI7_PC{pc_nums_str}_trg-{trg}.pkl')
    out_cii = pjoin(work_dir, f'PC_predict_ROI7_PC{pc_nums_str}_trg-{trg}.dlabel.nii')

    # loading
    pc_maps = nib.load(pc_file).get_fdata()[pc_indices]
    vis_mask = ~np.isnan(pc_maps[0])
    roi_names = []
    roi_sizes = []
    roi_name2key = {}
    if trg in ('MMP-vis3-L', 'MMP-vis3-R'):
        atlas_file = pjoin(work_dir, 'HCP-MMP1_visual-cortex3_size-300.dlabel.nii')
        reader = CiftiReader(atlas_file)
        atlas_map = reader.get_data()[0]
        mask = np.zeros_like(atlas_map, dtype=bool)
        for roi_name in get_rois(trg):
            roi_key = mmp_name2label[roi_name]
            roi_mask = atlas_map == roi_key
            mask[roi_mask] = True
            roi_names.append(roi_name)
            roi_sizes.append(np.sum(roi_mask))
            roi_name2key[roi_name] = roi_key
    elif trg in ('Wang2015-L', 'Wang2015-R'):
        reader = CiftiReader(wang2015_file)
        atlas = Atlas('Wang2015')
        atlas_map = atlas.maps[0]
        mask = np.zeros_like(atlas_map, dtype=bool)
        for roi_name in get_rois(trg):
            roi_key = atlas.roi2label[roi_name]
            roi_mask = atlas_map == roi_key
            roi_mask = np.logical_and(vis_mask, roi_mask)
            n_vtx = np.sum(roi_mask)
            if n_vtx > 0:
                mask[roi_mask] = True
                roi_names.append(roi_name)
                roi_sizes.append(n_vtx)
                roi_name2key[roi_name] = roi_key
    else:
        raise ValueError

    # prepare X
    X = pc_maps[:, mask].T
    y = atlas_map[mask]

    # prepare parameters for GridSearch
    Cs = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    param_grid = [
        {'classifier': [LogisticRegression(penalty='none', max_iter=10000)],
         'classifier__solver': ['saga', 'lbfgs']},
        {'classifier': [LogisticRegression(penalty='l2', max_iter=10000)],
         'classifier__solver': ['saga', 'lbfgs'],
         'classifier__C': Cs},
    ]

    # loop n_split train-test splits
    out_dict = {'model': [], 'train_score': [], 'test_score': []}
    out_map = np.zeros((1, mask.shape[0]), np.uint16)
    y_pred = np.zeros_like(y)
    skf = StratifiedKFold(n_split, shuffle=True, random_state=7)
    split_idx = 1
    for train_indices, test_indices in skf.split(X, y):
        time1 = time.time()
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        pipe = Pipeline([('preprocesser', StandardScaler()),
                         ('classifier', LogisticRegression())])
        grid = GridSearchCV(pipe, param_grid, cv=n_split, scoring='accuracy')
        grid.fit(X_train, y_train)
        y_pred[test_indices] = grid.predict(X_test)

        out_dict['model'].append(grid)
        out_dict['train_score'].append(grid.score(X_train, y_train))
        out_dict['test_score'].append(grid.score(X_test, y_test))
        print(f'Finished {split_idx}/{n_split}, '
              f'cost {time.time() - time1} seconds.')
        split_idx += 1
    out_map[0, mask] = y_pred
    out_dict['y_true'] = y
    out_dict['y_pred'] = y_pred
    out_dict['roi_name'] = roi_names
    out_dict['roi_size'] = roi_sizes
    out_dict['roi_name2key'] = roi_name2key

    # output
    pkl.dump(out_dict, open(out_pkl, 'wb'))
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab_old = reader.label_tables()[0]
    for k in np.unique(out_map):
        if trg in ('Wang2015-L', 'Wang2015-R') and k > 25:
            lbl_tab[k] = lbl_tab_old[k-25]
        lbl_tab[k] = lbl_tab_old[k]
    save2cifti(out_cii, out_map, CiftiReader(mmp_map_file).brain_models(),
               label_tables=[lbl_tab])


def grad_predict_ROI8(Hemi, grad_type, grad_nums, n_random, n_split=5):
    """
    用指定的gradient (grad_nums)预测MMP-vis3 ROIs
    0. 对于顶点数大于300的脑区，随机保留300个顶点。（重复n_random次）
    1. 以StratifiedKFold划分Train和Test（n_split），对于每次split，
    用GridSearchCV在Train上搜索超参数（划分Train和Validation的方法
    也是StratifiedKFold (n_split)），并用最优超参数在整个Train上训练模型，
    然后得到在Test上的预测。
    2. 所有split的基于Test的预测值合在一起就可以得到所有顶点的预测值。

    Args:
        grad_type (str):
        grad_nums (integers): 梯度成分的编号（从1开始）
            用于指定使用哪些梯度
        n_split (int, optional): Defaults to 5.
    """
    # settings
    pc_indices = [i - 1 for i in grad_nums]
    pc_nums_str = '+'.join(map(str, grad_nums))
    out_pkl = pjoin(work_dir, f'grad-predict-ROI8_{grad_type}-C{pc_nums_str}_rand-{n_random}_{Hemi}.pkl')

    # loading
    if grad_type == 'stru':
        pc_file = pjoin(
            anal_dir, 'decomposition/'
            f'HCPY-M+corrT_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    elif grad_type == 'Margulies':
        pc_file = pjoin(
            proj_dir, 'data/neuromaps-data/annotations/margulies2016/'
            'source-margulies2016_space-fsLR_den-32k_feature.dscalar.nii')
    elif grad_type == 'RSFC':
        pc_file = pjoin(
            anal_dir, 'decomposition/'
            f'S1200-grp-RSFC-z_MMP-vis3-{Hemi}2MMP-vis3-{Hemi}_zscore_PCA-comp.dscalar.nii')
    else:
        raise ValueError
    pc_maps = nib.load(pc_file).get_fdata()[pc_indices, :LR_count_32k]
    atlas = Atlas('HCP-MMP')
    roi_names = get_rois(f'MMP-vis3-{Hemi}')
    roi_name2mask = {}
    for roi_name in roi_names:
        roi_mask = atlas.get_mask(roi_name)[0]
        roi_name2mask[roi_name] = np.where(roi_mask)[0]

    # prepare parameters for GridSearch
    Cs = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    param_grid = [
        {'classifier': [LogisticRegression(penalty='none', max_iter=10000)],
         'classifier__solver': ['saga', 'lbfgs']},
        {'classifier': [LogisticRegression(penalty='l2', max_iter=10000)],
         'classifier__solver': ['saga', 'lbfgs'],
         'classifier__C': Cs},
    ]

    out_dict = {
        'roi_name': roi_names,
        'roi_size': [], 'roi_name2key': {},
        'train_score': np.zeros((n_random, n_split)),
        'test_score': np.zeros((n_random, n_split)),
        'y_true': [], 'y_pred': []}
    for rand_idx in range(n_random):
        mask = np.zeros(LR_count_32k, dtype=bool)
        for roi_name in roi_names:
            roi_mask_indices = roi_name2mask[roi_name]
            if len(roi_mask_indices) > 300:
                # 超过300个顶点的脑区随机留用300个顶点
                roi_mask_indices = np.random.choice(
                    roi_mask_indices, 300, False)
            mask[roi_mask_indices] = True
            if rand_idx == 0:
                out_dict['roi_size'].append(len(roi_mask_indices))
                out_dict['roi_name2key'][roi_name] = atlas.roi2label[roi_name]

        # prepare X
        X = pc_maps[:, mask].T
        y = atlas.maps[0, mask]

        # loop n_split train-test splits
        y_pred = np.zeros_like(y)
        skf = StratifiedKFold(n_split, shuffle=True, random_state=7)
        split_idx = 0
        for train_indices, test_indices in skf.split(X, y):
            time1 = time.time()
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            pipe = Pipeline([('preprocesser', StandardScaler()),
                            ('classifier', LogisticRegression())])
            grid = GridSearchCV(pipe, param_grid, cv=n_split, scoring='accuracy')
            grid.fit(X_train, y_train)
            y_pred[test_indices] = grid.predict(X_test)

            out_dict['train_score'][rand_idx, split_idx] = grid.score(X_train, y_train)
            out_dict['test_score'][rand_idx, split_idx] = grid.score(X_test, y_test)
            print(f'Finished {rand_idx+1}/{n_random}: {split_idx+1}/{n_split}, '
                  f'cost {time.time() - time1} seconds.')
            split_idx += 1
        out_dict['y_true'].append(y)
        out_dict['y_pred'].append(y_pred)

    # output
    pkl.dump(out_dict, open(out_pkl, 'wb'))


def grad_predict_ROI9(Hemi, grad_type, grad_nums, n_random, n_split=5):
    """
    用指定的gradient (grad_nums)预测MMP-vis3 ROIs
    0. 通过随机保留顶点使得所有脑区的顶点数与最小脑区一致。（重复n_random次）
    1. 以StratifiedKFold划分Train和Test（n_split），对于每次split，
    用GridSearchCV在Train上搜索超参数（划分Train和Validation的方法
    也是StratifiedKFold (n_split)），并用最优超参数在整个Train上训练模型，
    然后得到在Test上的预测。
    2. 所有split的基于Test的预测值合在一起就可以得到所有顶点的预测值。

    Args:
        grad_type (str):
        grad_nums (integers): 梯度成分的编号（从1开始）
            用于指定使用哪些梯度
        n_split (int, optional): Defaults to 5.
    """
    # settings
    pc_indices = [i - 1 for i in grad_nums]
    pc_nums_str = '+'.join(map(str, grad_nums))
    out_pkl = pjoin(work_dir, f'grad-predict-ROI9_{grad_type}-C{pc_nums_str}_rand-{n_random}_{Hemi}.pkl')

    # loading
    if grad_type == 'stru':
        pc_file = pjoin(
            anal_dir, 'decomposition/'
            f'HCPY-M+corrT_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    elif grad_type == 'Margulies':
        pc_file = pjoin(
            proj_dir, 'data/neuromaps-data/annotations/margulies2016/'
            'source-margulies2016_space-fsLR_den-32k_feature.dscalar.nii')
    elif grad_type == 'RSFC':
        pc_file = pjoin(
            anal_dir, 'decomposition/'
            f'S1200-grp-RSFC-z_MMP-vis3-{Hemi}2MMP-vis3-{Hemi}_zscore_PCA-comp.dscalar.nii')
    else:
        raise ValueError
    pc_maps = nib.load(pc_file).get_fdata()[pc_indices, :LR_count_32k]
    atlas = Atlas('HCP-MMP')
    roi_names = get_rois(f'MMP-vis3-{Hemi}')
    roi_name2mask = {}
    min_vtx_num = np.inf
    for roi_name in roi_names:
        roi_mask = atlas.get_mask(roi_name)[0]
        roi_name2mask[roi_name] = np.where(roi_mask)[0]
        n_vtx = len(roi_name2mask[roi_name])
        if n_vtx < min_vtx_num:
            min_vtx_num = n_vtx

    # prepare parameters for GridSearch
    Cs = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    param_grid = [
        {'classifier': [LogisticRegression(penalty='none', max_iter=10000)],
         'classifier__solver': ['saga', 'lbfgs']},
        {'classifier': [LogisticRegression(penalty='l2', max_iter=10000)],
         'classifier__solver': ['saga', 'lbfgs'],
         'classifier__C': Cs},
    ]

    out_dict = {
        'roi_name': roi_names,
        'roi_size': [], 'roi_name2key': {},
        'train_score': np.zeros((n_random, n_split)),
        'test_score': np.zeros((n_random, n_split)),
        'y_true': [], 'y_pred': []}
    for rand_idx in range(n_random):
        mask = np.zeros(LR_count_32k, dtype=bool)
        for roi_name in roi_names:
            roi_mask_indices = roi_name2mask[roi_name]
            if len(roi_mask_indices) > min_vtx_num:
                roi_mask_indices = np.random.choice(
                    roi_mask_indices, min_vtx_num, False)
            mask[roi_mask_indices] = True
            if rand_idx == 0:
                out_dict['roi_size'].append(len(roi_mask_indices))
                out_dict['roi_name2key'][roi_name] = atlas.roi2label[roi_name]

        # prepare X
        X = pc_maps[:, mask].T
        y = atlas.maps[0, mask]

        # loop n_split train-test splits
        y_pred = np.zeros_like(y)
        skf = StratifiedKFold(n_split, shuffle=True)
        split_idx = 0
        for train_indices, test_indices in skf.split(X, y):
            time1 = time.time()
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            pipe = Pipeline([('preprocesser', StandardScaler()),
                            ('classifier', LogisticRegression())])
            grid = GridSearchCV(pipe, param_grid, cv=n_split, scoring='accuracy')
            grid.fit(X_train, y_train)
            y_pred[test_indices] = grid.predict(X_test)

            out_dict['train_score'][rand_idx, split_idx] = grid.score(X_train, y_train)
            out_dict['test_score'][rand_idx, split_idx] = grid.score(X_test, y_test)
            print(f'Finished {rand_idx+1}/{n_random}: {split_idx+1}/{n_split}, '
                  f'cost {time.time() - time1} seconds.')
            split_idx += 1
        out_dict['y_true'].append(y)
        out_dict['y_pred'].append(y_pred)

    # output
    pkl.dump(out_dict, open(out_pkl, 'wb'))


def PC_predict_ALFF(Hemi, pc_nums, trg, n_split=5):
    """
    用指定的PC (pc_nums)预测(f)ALFF (trg)
    1. 以KFold划分Train和Test (n_split)，对于每次split，
        在Train上训练GLM模型，然后在Test上进行预测。
    2. 所有split的基于Test的预测值合在一起就可以得到所有顶点的预测值。
    3. 然后展示这个预测结果的脑图，对比MMP的ground truth脑图，算个相关。

    Args:
        Hemi (str): L or R
        pc_nums (integers): 主成分的编号（从1开始）
            用于指定使用哪些PC
        trg (str): ALFF or fALFF
        n_split (int, optional): Defaults to 5.
    """
    # settings
    vis_name = f'MMP-vis3-{Hemi}'
    pc_indices = [i - 1 for i in pc_nums]
    pc_file = pjoin(
        anal_dir, 'decomposition/'
        f'HCPY-M+corrT_{vis_name}_zscore1_PCA-subj.dscalar.nii')
    pc_nums_str = '+'.join(map(str, pc_nums))
    out_pkl = pjoin(work_dir, f'PC{pc_nums_str}_predict_{trg}_{Hemi}H.pkl')
    out_cii = pjoin(work_dir, f'PC{pc_nums_str}_predict_{trg}_{Hemi}H.dscalar.nii')

    # loading
    atlas = Atlas('HCP-MMP')
    vis_mask = atlas.get_mask(get_rois(vis_name))[0]
    pc_maps = nib.load(pc_file).get_fdata()[pc_indices]
    if trg == 'fALFF':
        trg_file = pjoin(anal_dir, 'AFF/HCPY-faff.dscalar.nii')
        trg_map = nib.load(trg_file).get_fdata()[0, :LR_count_32k]
    else:
        raise ValueError

    # prepare X, y
    X = pc_maps[:, vis_mask].T
    y = trg_map[vis_mask]

    # loop n_split train-test splits
    out_dict = {'model': [], 'train_score': [], 'test_score': []}
    out_map = np.ones((1, LR_count_32k)) * np.nan
    y_pred = np.zeros_like(y)
    kf = KFold(n_split, shuffle=True, random_state=7)
    split_idx = 1
    for train_indices, test_indices in kf.split(X, y):
        time1 = time.time()
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        pipe = Pipeline([('preprocesser', StandardScaler()),
                         ('regressor', LinearRegression())])
        pipe.fit(X_train, y_train)
        y_pred[test_indices] = pipe.predict(X_test)

        out_dict['model'].append(pipe)
        out_dict['train_score'].append(pipe.score(X_train, y_train))
        out_dict['test_score'].append(pipe.score(X_test, y_test))
        print(f'Finished {split_idx}/{n_split}, '
              f'cost {time.time() - time1} seconds.')
        split_idx += 1
    out_map[0, vis_mask] = y_pred
    out_dict['y_true'] = y
    out_dict['y_pred'] = y_pred

    # output
    pkl.dump(out_dict, open(out_pkl, 'wb'))
    save2cifti(out_cii, out_map, CiftiReader(mmp_map_file).brain_models())


def PC_predict_ALFF1(Hemi, pc_nums, trg, n_random, n_split=5):
    """
    用指定的PC (pc_nums)预测(f)ALFF (trg)
    1. 以KFold划分Train和Test (n_split)，对于每次split，
        在Train上训练GLM模型，然后在Test上进行预测。
    2. 所有split的基于Test的预测值合在一起就可以得到所有顶点的预测值。
    以上步骤重复n_random次

    Args:
        Hemi (str): L or R
        pc_nums (integers): 主成分的编号（从1开始）
            用于指定使用哪些PC
        trg (str): ALFF or fALFF
        n_split (int, optional): Defaults to 5.
    """
    # settings
    vis_name = f'MMP-vis3-{Hemi}'
    pc_indices = [i - 1 for i in pc_nums]
    pc_file = pjoin(
        anal_dir, 'decomposition/'
        f'HCPY-M+corrT_{vis_name}_zscore1_PCA-subj.dscalar.nii')
    pc_nums_str = '+'.join(map(str, pc_nums))
    out_pkl = pjoin(work_dir, f'PC{pc_nums_str}_predict_{trg}1_rand-{n_random}_{Hemi}H.pkl')

    # loading
    atlas = Atlas('HCP-MMP')
    vis_mask = atlas.get_mask(get_rois(vis_name))[0]
    pc_maps = nib.load(pc_file).get_fdata()[pc_indices]
    if trg == 'fALFF':
        trg_file = pjoin(anal_dir, 'AFF/HCPY-faff.dscalar.nii')
        trg_map = nib.load(trg_file).get_fdata()[0, :LR_count_32k]
    else:
        raise ValueError

    # prepare X, y
    X = pc_maps[:, vis_mask].T
    y = trg_map[vis_mask]

    # loop n_split train-test splits
    out_dict = {
        'train_score': np.zeros((n_random, n_split)),
        'test_score': np.zeros((n_random, n_split)),
        'y_true': y, 'y_pred': []}
    for rand_idx in range(n_random):
        y_pred = np.zeros_like(y)
        kf = KFold(n_split, shuffle=True)
        split_idx = 0
        for train_indices, test_indices in kf.split(X, y):
            time1 = time.time()
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            pipe = Pipeline([('preprocesser', StandardScaler()),
                            ('regressor', LinearRegression())])
            pipe.fit(X_train, y_train)
            y_pred[test_indices] = pipe.predict(X_test)

            out_dict['train_score'][rand_idx, split_idx] = pipe.score(X_train, y_train)
            out_dict['test_score'][rand_idx, split_idx] = pipe.score(X_test, y_test)
            print(f'Finished {rand_idx+1}/{n_random} {split_idx+1}/{n_split}, '
                  f'cost {time.time() - time1} seconds.')
            split_idx += 1
        out_dict['y_pred'].append(y_pred)

    # output
    pkl.dump(out_dict, open(out_pkl, 'wb'))


if __name__ == '__main__':
    # PC12_predict_ROI1()
    # PC12_predict_ROI2()
    # PC12_predict_ROI3()

    # ===PC_predict_ROI4===
    # PC_predict_ROI4(
    #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #     pc_nums=[i for i in range(1, 11) if i != 2], mask_name='MMP-vis3-R', n_split=5)

    # for i in range(3, 11):
    #     PC_predict_ROI4(
    #         pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #         pc_nums=[1, i], mask_name='MMP-vis3-R', n_split=5)
    # for i in range(3, 11):
    #     PC_predict_ROI4(
    #         pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-L_zscore1_PCA-subj.dscalar.nii'),
    #         pc_nums=[1, i], mask_name='MMP-vis3-L', n_split=5)

    # for mask_name in [
    #     'MMP-vis3-R-early', 'MMP-vis3-R-dorsal',
    #     'MMP-vis3-R-lateral', 'MMP-vis3-R-ventral',
        # 'MMP-vis3-R-forefront', 'MMP-vis3-R-ventral_ex(forefront)',
        # 'MMP-vis3-R',
        # 'MMP-vis3-R_ex(V1~4)',
    # ]:
        # PC_predict_ROI4(
        #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
        #     pc_nums=[1], mask_name=mask_name, n_split=5)
        # PC_predict_ROI4(
        #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
        #     pc_nums=[2], mask_name=mask_name, n_split=5)
        # PC_predict_ROI4(
        #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
        #     pc_nums=[1, 2], mask_name=mask_name, n_split=5)
        # for i in range(3, 6):
        # for i in range(6, 11):
        #     PC_predict_ROI4(
        #         pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
        #         pc_nums=[1, i], mask_name=mask_name, n_split=5)

    # for mask_name in [
    #     # 'MMP-vis3-L',
    #     'MMP-vis3-L-early', 'MMP-vis3-L-dorsal',
    #     'MMP-vis3-L-lateral', 'MMP-vis3-L-ventral',
    # ]:
        # PC_predict_ROI4(
        #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-L_zscore1_PCA-subj.dscalar.nii'),
        #     pc_nums=[1], mask_name=mask_name, n_split=5)
        # PC_predict_ROI4(
        #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-L_zscore1_PCA-subj.dscalar.nii'),
        #     pc_nums=[2], mask_name=mask_name, n_split=5)
        # PC_predict_ROI4(
        #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-L_zscore1_PCA-subj.dscalar.nii'),
        #     pc_nums=[1, 2], mask_name=mask_name, n_split=5)
        # for i in range(3, 6):
        # for i in range(6, 11):
        #     PC_predict_ROI4(
        #         pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-L_zscore1_PCA-subj.dscalar.nii'),
        #         pc_nums=[1, i], mask_name=mask_name, n_split=5)

    # ===PC_predict_ROI5===
    # for i in range(3, 11):
    #     PC_predict_ROI5(
    #         pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #         pc_nums=[1, i], mask_name='MMP-vis3-R', n_split=5)

    # for mask_name in [
    #     'MMP-vis3-R-early', 'MMP-vis3-R-dorsal', 'MMP-vis3-R-lateral',
    #     'MMP-vis3-R-ventral', 'MMP-vis3-R-forefront',
    #     'MMP-vis3-R-ventral_ex(forefront)', 'MMP-vis3-R',
    #     'MMP-vis3-R_ex(V1~4)',
    # ]:
    #     PC_predict_ROI5(
    #         pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #         pc_nums=[1], mask_name=mask_name, n_split=5)
    #     PC_predict_ROI5(
    #         pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #         pc_nums=[2], mask_name=mask_name, n_split=5)
    #     PC_predict_ROI5(
    #         pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #         pc_nums=[1, 2], mask_name=mask_name, n_split=5)

    # PC_predict_ROI6(
    #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #     trg='MPM', pc_nums=[1], mask_name='R_V1~4', n_split=5)
    # PC_predict_ROI6(
    #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #     trg='MPM', pc_nums=[1], mask_name='MMP-vis3-R_ex(V1~4)', n_split=5)
    # PC_predict_ROI6(
    #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #     trg='MPM', pc_nums=[2], mask_name='R_V1~4', n_split=5)
    # PC_predict_ROI6(
    #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #     trg='MPM', pc_nums=[2], mask_name='MMP-vis3-R_ex(V1~4)', n_split=5)

    # PC_predict_ROI6(
    #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #     trg='animate', pc_nums=[1], mask_name='R_V1~4', n_split=5)
    # PC_predict_ROI6(
    #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #     trg='animate', pc_nums=[1], mask_name='MMP-vis3-R_ex(V1~4)', n_split=5)
    # PC_predict_ROI6(
    #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #     trg='animate', pc_nums=[2], mask_name='R_V1~4', n_split=5)
    # PC_predict_ROI6(
    #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #     trg='animate', pc_nums=[2], mask_name='MMP-vis3-R_ex(V1~4)', n_split=5)

    # PC_predict_ROI6(
    #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #     trg='count', pc_nums=[1], mask_name='R_V1~4', n_split=5)
    # PC_predict_ROI6(
    #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #     trg='count', pc_nums=[1], mask_name='MMP-vis3-R_ex(V1~4)', n_split=5)
    # PC_predict_ROI6(
    #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #     trg='count', pc_nums=[2], mask_name='R_V1~4', n_split=5)
    # PC_predict_ROI6(
    #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #     trg='count', pc_nums=[2], mask_name='MMP-vis3-R_ex(V1~4)', n_split=5)

    # prepare_visual_ROIs()
    # for trg in [
    #     # 'MMP-vis3-R',
    #     'Wang2015-R'
    # ]:
    #     PC_predict_ROI7(
    #         pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #         pc_nums=[1], trg=trg, n_split=5)
    #     PC_predict_ROI7(
    #         pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #         pc_nums=[2], trg=trg, n_split=5)
    #     PC_predict_ROI7(
    #         pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #         pc_nums=[1, 2], trg=trg, n_split=5)

    # for trg in [
    #     'MMP-vis3-L', 'Wang2015-L'
    # ]:
    #     PC_predict_ROI7(
    #         pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-L_zscore1_PCA-subj.dscalar.nii'),
    #         pc_nums=[1], trg=trg, n_split=5)
    #     PC_predict_ROI7(
    #         pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-L_zscore1_PCA-subj.dscalar.nii'),
    #         pc_nums=[2], trg=trg, n_split=5)
    #     PC_predict_ROI7(
    #         pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+corrT_MMP-vis3-L_zscore1_PCA-subj.dscalar.nii'),
    #         pc_nums=[1, 2], trg=trg, n_split=5)

    # grad_predict_ROI8(Hemi='R', grad_type='stru', grad_nums=[1], n_random=10, n_split=5)
    # grad_predict_ROI8(Hemi='R', grad_type='stru', grad_nums=[2], n_random=10, n_split=5)
    # grad_predict_ROI8(Hemi='R', grad_type='stru', grad_nums=[1, 2], n_random=10, n_split=5)
    # grad_predict_ROI8(Hemi='R', grad_type='Margulies', grad_nums=[1, 2], n_random=10, n_split=5)
    # grad_predict_ROI8(Hemi='R', grad_type='RSFC', grad_nums=[1, 2], n_random=10, n_split=5)

    # grad_predict_ROI8(Hemi='R', grad_type='stru', grad_nums=[1], n_random=100, n_split=5)
    # grad_predict_ROI8(Hemi='R', grad_type='stru', grad_nums=[2], n_random=100, n_split=5)
    # grad_predict_ROI8(Hemi='R', grad_type='stru', grad_nums=[1, 2], n_random=100, n_split=5)
    # grad_predict_ROI8(Hemi='R', grad_type='Margulies', grad_nums=[1, 2], n_random=100, n_split=5)
    # grad_predict_ROI8(Hemi='R', grad_type='RSFC', grad_nums=[1, 2], n_random=100, n_split=5)
    # grad_predict_ROI8(Hemi='L', grad_type='stru', grad_nums=[1], n_random=100, n_split=5)
    # grad_predict_ROI8(Hemi='L', grad_type='stru', grad_nums=[2], n_random=100, n_split=5)
    # grad_predict_ROI8(Hemi='L', grad_type='stru', grad_nums=[1, 2], n_random=100, n_split=5)
    # grad_predict_ROI8(Hemi='L', grad_type='Margulies', grad_nums=[1, 2], n_random=100, n_split=5)
    # grad_predict_ROI8(Hemi='L', grad_type='RSFC', grad_nums=[1, 2], n_random=100, n_split=5)

    grad_predict_ROI9(Hemi='R', grad_type='stru', grad_nums=[1], n_random=100, n_split=5)
    grad_predict_ROI9(Hemi='R', grad_type='stru', grad_nums=[2], n_random=100, n_split=5)
    grad_predict_ROI9(Hemi='R', grad_type='stru', grad_nums=[1, 2], n_random=100, n_split=5)
    # grad_predict_ROI9(Hemi='R', grad_type='Margulies', grad_nums=[1, 2], n_random=100, n_split=5)
    # grad_predict_ROI9(Hemi='R', grad_type='RSFC', grad_nums=[1, 2], n_random=100, n_split=5)
    grad_predict_ROI9(Hemi='L', grad_type='stru', grad_nums=[1], n_random=100, n_split=5)
    grad_predict_ROI9(Hemi='L', grad_type='stru', grad_nums=[2], n_random=100, n_split=5)
    grad_predict_ROI9(Hemi='L', grad_type='stru', grad_nums=[1, 2], n_random=100, n_split=5)
    # grad_predict_ROI9(Hemi='L', grad_type='Margulies', grad_nums=[1, 2], n_random=100, n_split=5)
    # grad_predict_ROI9(Hemi='L', grad_type='RSFC', grad_nums=[1, 2], n_random=100, n_split=5)

    # PC_predict_ALFF(Hemi='R', pc_nums=[1], trg='fALFF', n_split=5)
    # PC_predict_ALFF(Hemi='R', pc_nums=[2], trg='fALFF', n_split=5)
    # PC_predict_ALFF(Hemi='R', pc_nums=[1, 2], trg='fALFF', n_split=5)
    # PC_predict_ALFF(Hemi='L', pc_nums=[1], trg='fALFF', n_split=5)
    # PC_predict_ALFF(Hemi='L', pc_nums=[2], trg='fALFF', n_split=5)
    # PC_predict_ALFF(Hemi='L', pc_nums=[1, 2], trg='fALFF', n_split=5)

    # PC_predict_ALFF1(Hemi='R', pc_nums=[1], trg='fALFF', n_random=100, n_split=5)
    # PC_predict_ALFF1(Hemi='R', pc_nums=[2], trg='fALFF', n_random=100, n_split=5)
    # PC_predict_ALFF1(Hemi='R', pc_nums=[1, 2], trg='fALFF', n_random=100, n_split=5)
    # PC_predict_ALFF1(Hemi='L', pc_nums=[1], trg='fALFF', n_random=100, n_split=5)
    # PC_predict_ALFF1(Hemi='L', pc_nums=[2], trg='fALFF', n_random=100, n_split=5)
    # PC_predict_ALFF1(Hemi='L', pc_nums=[1, 2], trg='fALFF', n_random=100, n_split=5)
