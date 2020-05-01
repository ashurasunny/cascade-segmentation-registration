import os
import numpy as np
import torch
from utils import data_augmenter, rescale_intensity, convert_to_one_hot
import random


def make_dataset(root):
    imgs = []
    for folder in os.listdir(root):
        folder_root = os.path.join(root, folder)
        for f in os.listdir(folder_root):
            if 'gt' in f or 'ES' in f:
                continue
            gt_f = f.replace('.npy', '_gt.npy')
            es_f = f.replace('ED', 'ES')
            gt_es_f = es_f.replace('.npy', '_gt.npy')
            imgs.append([os.path.join(folder_root, f), os.path.join(folder_root, gt_f),
                         os.path.join(folder_root, es_f), os.path.join(folder_root, gt_es_f)])
    return imgs


def get_epoch_batch(data_list, batch_size, iteration, idx, image_size=192, data_augmentation=False,
                    shift=0.0, rotate=0.0, scale=0.0, intensity=0.0, flip=False, norm=True, aug_rate=0.5):
    eds, ed_gts, ess, es_gts = [], [], [], []

    for i in range(iteration * batch_size, (iteration + 1) * batch_size):

        es_name, es_gt_name, ed_name, ed_gt_name = data_list[idx[i]]
        if np.random.uniform() > 0.5:
            temp = es_name
            es_name = ed_name
            ed_name = temp

            temp = es_gt_name
            es_gt_name = ed_gt_name
            ed_gt_name = temp
        if os.path.exists(es_name) and os.path.exists(es_gt_name):
            # print('  Select {0} {1}'.format(image_name, label_name))

            # Read image and label
            # print(es_name)
            es = np.load(es_name)
            es_gt = np.load(es_gt_name)
            ed = np.load(ed_name)
            ed_gt = np.load(ed_gt_name)

            # Handle exceptions
            if es.shape != es_gt.shape:
                print('Error: mismatched size, image.shape = {0}, label.shape = {1}'.format(image.shape, label.shape))
                print('Skip {0}, {1}'.format(es_name, es_gt_name))
                continue

            if es.max() < 1e-6:
                print('Error: blank image, image.max = {0}'.format(es.max()))
                print('Skip {0} {1}'.format(es_name, es_gt_name))
                continue

            # Append the image slices to the batch
            # Use list for appending, which is much faster than numpy array

            Z = es.shape[0]

            if Z > 8:
                r = Z - 8
                start = random.randint(0, r)
                for z in range(start, start + 8):
                    temp1 = es[z, :, :]
                    temp2 = ed[z, :, :]
                    if norm:
                        temp1 = rescale_intensity(temp1, (0.5, 99.5))
                        temp2 = rescale_intensity(temp2, (0.5, 99.5))
                    ess += [temp1]
                    eds += [temp2]
                    es_gts += [es_gt[z, :, :]]
                    ed_gts += [ed_gt[z, :, :]]
            else:
                for z in range(Z):
                    temp1 = es[z, :, :]
                    temp2 = ed[z, :, :]
                    if norm:
                        temp1 = rescale_intensity(temp1, (0.5, 99.5))
                        temp2 = rescale_intensity(temp2, (0.5, 99.5))
                    ess += [temp1]
                    eds += [temp2]
                    es_gts += [es_gt[z, :, :]]
                    ed_gts += [ed_gt[z, :, :]]

    # Convert to a numpy array
    ESs = np.array(ess, dtype=np.float32)
    ES_gts = np.array(es_gts, dtype=np.float32)
    EDs = np.array(eds, dtype=np.float32)
    ED_gts = np.array(ed_gts, dtype=np.float32)

    # Add the channel dimension
    # tensorflow by default assumes NHWC format (batch_size, 128, 128, 1)
    ESs = np.expand_dims(ESs, axis=3)
    EDs = np.expand_dims(EDs, axis=3)
    # Perform data augmentation
    if data_augmentation  and np.random.uniform() > aug_rate:
        ESs, ES_gts = data_augmenter(ESs, ES_gts, shift=shift, rotate=rotate,
                                     scale=scale, intensity=intensity, flip=flip)
        EDs, ED_gts = data_augmenter(EDs, ED_gts, shift=shift, rotate=rotate,
                                        scale=scale, intensity=intensity, flip=flip)

    ESs_onehot = convert_to_one_hot(ES_gts).astype(np.float32)
    ESs_onehot = ESs_onehot.transpose([1, 0, 2, 3])
    ESs_onehot = torch.from_numpy(ESs_onehot)
    EDs_onehot = convert_to_one_hot(ED_gts).astype(np.float32)
    EDs_onehot = EDs_onehot.transpose([1, 0, 2, 3])
    EDs_onehot = torch.from_numpy(EDs_onehot)

    ES_M = ESs.copy()
    ES_M[ES_gts == 0] = 0
    ED_M = EDs.copy()
    ED_M[ED_gts == 0] = 0

    ESs = torch.from_numpy(ESs.transpose((0, 3, 1, 2)))
    ES_M = torch.from_numpy(ES_M.transpose((0, 3, 1, 2)))
    EDs = torch.from_numpy(EDs.transpose((0, 3, 1, 2)))
    ED_M = torch.from_numpy(ED_M.transpose((0, 3, 1, 2)))


    return {'ED': EDs, 'ED_gt': EDs_onehot,'ED_M': ED_M, 'ES': ESs, 'ES_gt': ESs_onehot,'ES_M': ES_M}


if __name__ == '__main__':
    data_list = make_dataset('/run/media/xxw993/Ashura/ACDC_XI/training_crop')
    idx = range(len(data_list))
    sample = get_epoch_batch(data_list, 1, 1, idx)
    x, y = sample
    print(x.size(), y.size())