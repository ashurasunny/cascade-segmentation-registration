import os
import nibabel as nib
import numpy as np
import cv2
from skimage.transform import resize
import SimpleITK as sitk
from scipy import ndimage
from skimage.morphology import label
from scipy.ndimage import distance_transform_edt as distance
import torch
from torch import Tensor
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union

def postprocess_prediction(seg):
    tmp = convert_to_one_hot(seg)
    vals = np.unique(seg)
    results = []
    for i in range(len(tmp)):
        temp = largest_connected_componets(tmp[i])
        if i == 1:
            temp = open_close(temp,3)
        else:
            temp = open_close(temp, 1)
        results.append(temp[None])
    seg = vals[np.vstack(results).argmax(0)]
    # seg = open_close(seg)
    return seg

def largest_connected_componets(seg):
    # basically look for connected components and choose the largest one, delete everything else
    mask = seg != 0
    lbls = label(mask, 8)
    lbls_sizes = [np.sum(lbls == i) for i in np.unique(lbls)]
    largest_region = np.argmax(lbls_sizes[1:]) + 1
    seg[lbls != largest_region] = 0
    return seg

def open_close(img, iteration=1):
    z = img.shape[0]
    result = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    for i in range(z):
        temp = img[i,:,:]
        temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel, iterations=1)
        temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel, iterations=1)
        temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel, iterations=iteration)
        temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel, iterations=iteration)
        result.append(temp)
    return np.array(result)

# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def convert_to_one_hot(seg):
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res


def data_augmenter(image, label, shift, rotate, scale, intensity, flip):
    """
        Online data augmentation
        Perform affine transformation on image and label,
        which are 4D tensor of shape (N, H, W, C) and 3D tensor of shape (N, H, W).
    """
    image2 = np.zeros(image.shape, dtype=np.float32)
    label2 = np.zeros(label.shape, dtype=label.dtype)

    for i in range(image.shape[0]):
        # For each image slice, generate random affine transformation parameters
        # using the Gaussian distribution
        shift_val = [np.clip(np.random.normal(), -3, 3) * shift,
                     np.clip(np.random.normal(), -3, 3) * shift]
        rotate_val = np.clip(np.random.normal(), -3, 3) * rotate
        scale_val = 1 + np.clip(np.random.normal(), -3, 3) * scale
        intensity_val = 1 + np.clip(np.random.normal(), -3, 3) * intensity

        # Apply the affine transformation (rotation + scale + shift) to the image
        row, col = image.shape[1:3]
        M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_val, 1.0 / scale_val)
        M[:, 2] += shift_val
        for c in range(image.shape[3]):
            image2[i, :, :, c] = ndimage.interpolation.affine_transform(image[i, :, :, c], M[:, :2], M[:, 2], order=1)

        # Apply the affine transformation (rotation + scale + shift) to the label map
        label2[i, :, :] = ndimage.interpolation.affine_transform(label[i, :, :], M[:, :2], M[:, 2], order=0)

        # Apply intensity variation
        image2[i] *= intensity_val

        # Apply random horizontal or vertical flipping
        if flip:
            if np.random.uniform() >= 0.5:
                image2[i] = image2[i, ::-1, :, :]
                label2[i] = label2[i, ::-1, :]
            else:
                image2[i] = image2[i, :, ::-1, :]
                label2[i] = label2[i, :, ::-1]
    return image2, label2


def resize_image(image, old_spacing, new_spacing, order=3):
    new_shape = (int(np.round(old_spacing[0] / new_spacing[0] * float(image.shape[0]))),
                 int(np.round(old_spacing[1] / new_spacing[1] * float(image.shape[1]))),
                 int(np.round(old_spacing[2] / new_spacing[2] * float(image.shape[2]))))
    return resize(image, new_shape, order=order, mode='edge')


def resize_nifti(root, outputroot, spacing_target):
    for folder in os.listdir(root):
        folder_root = os.path.join(root, folder)
        output_folder_root = os.path.join(outputroot, folder)
        if not os.path.exists(output_folder_root):
            os.mkdir(output_folder_root)
        for nii in os.listdir(folder_root):
            print(folder, nii)
            if 'nii.gz' not in nii or '4d' in nii:
                continue
            itk_image = sitk.ReadImage(os.path.join(folder_root, nii))
            spacing = np.array(itk_image.GetSpacing())[[2, 1, 0]]
            image = sitk.GetArrayFromImage(itk_image).astype(float)

            # keep z spacing, do not modify
            spacing_target = list(spacing_target)
            spacing_target[0] = spacing[0]

            if 'gt' in nii:
                tmp = convert_to_one_hot(image)
                vals = np.unique(image)
                results = []
                for i in range(len(tmp)):
                    results.append(resize_image(tmp[i].astype(float), spacing, spacing_target, 1)[None])
                image = vals[np.vstack(results).argmax(0)]
            else:
                image = resize_image(image, spacing, spacing_target).astype(np.float32)
                image -= image.mean()
                image /= image.std()
            np.save(os.path.join(output_folder_root, nii.replace('.nii.gz', '.npy')), image)

def my_round(input):
    int_part = input.astype(np.int32)
    float_part = input - int_part
    input[float_part<0.5] = int_part[float_part<0.5]
    input[float_part>=0.5] = int_part[float_part>=0.5]+1
    return input

def rescale_intensity(image, thres=(1.0, 99.0)):
    """ Rescale the image intensity to the range of [0, 1] """
    image = image.astype(np.float32)
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    # image2 = (image - image.min()) / (image.max() - image.min())
    return image2

def rescale_intensity_slice(image, thres):
    Z = image.shape[0]
    output = []
    for z in range(Z):
        output.append(rescale_intensity(image[z, :, :], thres))
    return np.array(output)

if __name__ == '__main__':
    data = np.load('/run/media/xxw993/Ashura/ACDC_XI/training_crop/patient001/crop1_ED_gt.npy')
    # data = test.get_data()
    oh = convert_to_one_hot(data)
    dist = one_hot2dist(oh)
    dist = np.abs(dist)
    dist = 255 * dist / dist.max()
    cv2.imwrite('./test.png', dist[0,5,:,:])
    print('hello world')


