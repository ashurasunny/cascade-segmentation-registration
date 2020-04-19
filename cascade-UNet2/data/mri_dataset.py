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
            if 'gt' in f:
                continue
            gt_f = f.replace('.npy', '_gt.npy')
            imgs.append([os.path.join(folder_root, f), os.path.join(folder_root, gt_f)])
    return imgs


def get_epoch_batch(data_list, batch_size, iteration, idx, image_size=192, data_augmentation=False,
                    shift=0.0, rotate=0.0, scale=0.0, intensity=0.0, flip=False, norm=True):
    images, labels = [], []

    for i in range(iteration * batch_size, (iteration + 1) * batch_size):

        image_name, label_name = data_list[idx[i]]

        if os.path.exists(image_name) and os.path.exists(label_name):
            # print('  Select {0} {1}'.format(image_name, label_name))

            # Read image and label
            image = np.load(image_name)
            label = np.load(label_name)

            # Handle exceptions
            if image.shape != label.shape:
                print('Error: mismatched size, image.shape = {0}, label.shape = {1}'.format(image.shape, label.shape))
                print('Skip {0}, {1}'.format(image_name, label_name))
                continue

            if image.max() < 1e-6:
                print('Error: blank image, image.max = {0}'.format(image.max()))
                print('Skip {0} {1}'.format(image_name, label_name))
                continue

            # Append the image slices to the batch
            # Use list for appending, which is much faster than numpy array

            Z = image.shape[0]

            if Z > 8:
                r = Z - 8
                start = random.randint(0, r)
                for z in range(start, start + 8):
                    temp = image[z, :, :]
                    if norm:
                        temp = rescale_intensity(temp, (0.5, 99.5))
                    images += [temp]
                    labels += [label[z, :, :]]
            else:
                for z in range(Z):
                    temp = image[z, :, :]
                    if norm:
                        temp = rescale_intensity(temp, (0.5, 99.5))
                    images += [temp]
                    labels += [label[z, :, :]]

    # Convert to a numpy array

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    # Add the channel dimension
    # tensorflow by default assumes NHWC format (batch_size, 128, 128, 1)
    images = np.expand_dims(images, axis=3)

    # Perform data augmentation
    if data_augmentation:
        images, labels = data_augmenter(images, labels, shift=shift, rotate=rotate,
                                        scale=scale, intensity=intensity, flip=flip)
    labels_onehot = convert_to_one_hot(labels).astype(np.float32)
    labels_onehot = labels_onehot.transpose([1, 0, 2, 3])
    labels_onehot = torch.from_numpy(labels_onehot)
    M = images.copy()
    M[labels == 0] = 0
    # images2 = np.concatenate([images, M], axis=3)
    # images2 = torch.from_numpy(images.transpose((0, 3, 1, 2)))
    images = torch.from_numpy(images.transpose((0, 3, 1, 2)))
    labels = np.expand_dims(labels, axis=1)

    labels = torch.from_numpy(labels)
    M = torch.from_numpy(M.transpose((0, 3, 1, 2)))


    return {'A': images, 'B': labels_onehot,'M': M}


if __name__ == '__main__':
    data_list = make_dataset('/run/media/xxw993/Ashura/ACDC_XI/training_crop')
    idx = range(len(data_list))
    sample = get_epoch_batch(data_list, 1, 1, idx)
    x, y = sample
    print(x.size(), y.size())