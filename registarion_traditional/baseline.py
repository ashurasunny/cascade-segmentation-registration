import numpy as np
import cv2, os
import torch
import torch_dct as tdct
import torch.nn.functional as F
import SimpleITK as sitk
import math
from skimage.transform import resize

import time
from scipy.fftpack import dct, idct


PI = 3.1415926

def dct2(block):
    block = block.data.cpu().numpy()
    block = dct(dct(block.T, norm='ortho').T, norm='ortho')
    return torch.from_numpy(block).cuda()

def idct2(block):
    block = block.data.cpu().numpy()
    block = idct(idct(block.T, norm='ortho').T, norm='ortho')
    return torch.from_numpy(block).cuda()

def centralFiniteDifference(tensor):
    # tensor = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])


    t1 = tensor.clone()
    t2 = tensor.clone()
    t3 = tensor.clone()
    t4 = tensor.clone()

    t1[:, 1:] = tensor[:, :-1]
    t2[:, :-1] = tensor[:, 1:]
    t3[1:, :] = tensor[:-1, :]
    t4[:-1, :] = tensor[1:, :]




    dx = (t2-t1) / 2.0
    dy = (t4-t3) / 2.0
    return dx, dy


def flow_to_hsv(opt_flow, max_mag=1, white_bg=True):
    """
    Encode optical flow to HSV.

    Args:
        opt_flow: 2D optical flow in (dx, dy) encoding, shape (H, W, 2)
        max_mag: flow magnitude will be normalised to [0, max_mag]

    Returns:
        hsv_flow_rgb: HSV encoded flow converted to RGB (for visualisation), same shape as input

    """
    # convert to polar coordinates
    mag, ang = cv2.cartToPolar(opt_flow[0,:,:], opt_flow[1,:,:])

    # hsv encoding
    hsv_flow = np.zeros((opt_flow.shape[1], opt_flow.shape[2], 3))
    hsv_flow[..., 0] = ang*180/np.pi/2  # hue = angle
    hsv_flow[..., 1] = 255.0  # saturation = 255
    mag = 0.3 * mag / max_mag
    mag[mag>1]=1
    hsv_flow[..., 2] = 255.0 * mag
    # (wrong) hsv_flow[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # convert hsv encoding to rgb for visualisation
    # ([..., ::-1] converts from BGR to RGB)
    hsv_flow_rgb = cv2.cvtColor(hsv_flow.astype(np.uint8), cv2.COLOR_HSV2BGR)[..., ::-1]
    hsv_flow_rgb = hsv_flow_rgb.astype(np.uint8)

    if white_bg:
        hsv_flow_rgb = 255 - hsv_flow_rgb
    return hsv_flow_rgb


def warp(source, offset_w, offset_h, interp='bilinear'):
    # generate standard mesh grid
    h, w = source.shape[-2:]
    grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h), torch.linspace(-1, 1, w)])

    # stop autograd from calculating gradients on standard grid line
    grid_h = grid_h.cuda()
    grid_w = grid_w.cuda()

    # (N, 1, H, W) -> (N, H, W)
    #    print('################### offset_h: {}'.format(offset_h.shape))
    # offset_h = offset_h.squeeze(1)
    # offset_w = offset_w.squeeze(1)
    #    print('################### offset_h: {}'.format(offset_h.shape))

    # (h,w) + (N, h, w) add by broadcasting
    grid_h = grid_h + offset_h * 2 / h
    grid_w = grid_w + offset_w * 2 / w

    #    print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
    #    print(source.shape)

    # each pair of coordinates on deformed grid is using x-y order,
    # i.e. (column_num, row_num)
    # as required by the the grid_sample() function
    #    print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
    #    print(grid_h.shape)
    deformed_grid = torch.stack((grid_w, grid_h), 3)  # shape (N, H, W, 2)
    #    print(deformed_grid.shape)
    deformed_image = F.grid_sample(source, deformed_grid, mode=interp)
    #    print(deformed_image.shape)
    return deformed_image


def optimize(ux0, uy0, im1, im2, maxIter, lambda_r, to1, theta, device):
    eps = 0.0000001
    Ix, Iy = centralFiniteDifference(im1)
    It = im1 - im2

    a11 = Ix*Ix
    a12 = Ix*Iy
    a22 = Iy*Iy


    t1 = Ix * (It - Ix * ux0 - Iy * uy0)
    t2 = Iy * (It - Ix * ux0 - Iy * uy0)

    h,w = im1.size()

    vx = torch.zeros((h,w)).to(device).double()
    vy = torch.zeros((h,w)).to(device).double()
    bx = torch.zeros((h, w)).to(device).double()
    by = torch.zeros((h, w)).to(device).double()
    ux = torch.zeros((h, w)).to(device).double()
    uy = torch.zeros((h, w)).to(device).double()

    X, Y = torch.meshgrid([torch.arange(0, h), torch.arange(0, w)])
    # G = 2 * (torch.cos(PI * X / w + PI * Y / h) - 2)
    # G = G.to(device).double()

    X, Y = torch.meshgrid(torch.linspace(0, h - 1, h), torch.linspace(0, w - 1, w))
    X, Y = X.cuda(), Y.cuda()
    G = torch.cos(math.pi * X / h) + torch.cos(math.pi * Y / w) - 2
    # G = G.unsqueeze(0).repeat(N, 1, 1, 1)

    for i in range(maxIter):
        tempx = ux
        tempy = uy

        h1 = theta * (vx - bx) - t1
        h2 = theta * (vy - by) - t2


        ux = ((a22 + theta) * h1 - a12 * h2) / ((a11 + theta) * (a22 + theta) - a12 * a12)
        uy = ((a11 + theta) * h2 - a12 * h1) / ((a11 + theta) * (a22 + theta) - a12 * a12)


        # vx = (idct2(dct2(theta * (ux + bx)) / (theta + lambda_r * G * G)))
        # vy = (idct2(dct2(theta * (uy + by)) / (theta + lambda_r * G * G)))

        vx = (tdct.idct_2d(tdct.dct_2d(theta * (ux + bx)) / (theta + lambda_r * G * G)))
        vy = (tdct.idct_2d(tdct.dct_2d(theta * (uy + by)) / (theta + lambda_r * G * G)))



        bx = bx + ux - vx
        by = by + uy - vy

        # t1 = Ix * (It - Ix * ux - Iy * uy)
        # t2 = Iy * (It - Ix * ux - Iy * uy)

        stopx = torch.sum(torch.abs(ux - tempx)) / (torch.sum(torch.abs(tempx)) + eps)
        stopy = torch.sum(torch.abs(uy - tempy)) / (torch.sum(torch.abs(tempy)) + eps)
        # print(i, stopx, stopy)
        if stopx < to1 and stopy < to1:
            print('iterate {} times, stop due to converge to tolerance'.format(i))
            break

    if i == maxIter -1:
        print('iterate {} times, stop due to reach max iteration'.format(i))
    return ux, uy


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


if __name__ == '__main__':
    # root = '/data/private/xxw993/data/training50/slice/patient002'
    # ed = np.load(os.path.join(root, 'crop1_ED.npy'))[3,:,:]
    # es = np.load(os.path.join(root, 'crop1_ES.npy'))[3,:,:]




    niiroot = '/data/private/xxw993/PycharmProjects/clear_version_eccv/crop/sa_crop.nii.gz'
    img = sitk.ReadImage(niiroot)
    img = sitk.GetArrayFromImage(img)
    target = img[0,:,:,:]
    source = img[14,:,:,:]

    target = rescale_intensity(target)
    source = rescale_intensity(source)
    source = source[4,:,:]
    target = target[4,:,:]
    source = np.pad(source, [4,4]).transpose([1,0])
    target = np.pad(target, [4,4]).transpose([1,0])
    device = 'cuda'
    h,w = source.shape

    level = [4, 2, 1]
    ux = torch.zeros((h//level[0],w//level[0])).to(device).double()
    uy = torch.zeros((h//level[0], w//level[0])).to(device).double()

    to1 = 1e-5
    lambda_ = 0.1
    theta = 0.1
    maxIter = 200000
    taylor = 10

    a = time.time()

    for l in level:
        s = resize(source, (h//l, w//l), 2)
        t = resize(target, (h//l, w//l), 2)
        s = torch.from_numpy(s).to(device).double()
        t = torch.from_numpy(t).to(device).double()

        print(ux.size(), s.size())
        diff_t = []
        for i_t in range(taylor):


            print('taylor:', i_t)
            ux, uy = optimize(ux, uy, s, t, maxIter, lambda_, to1, theta, device)
            warped_s = warp(s.unsqueeze(0).unsqueeze(0), ux.unsqueeze(0), uy.unsqueeze(0))
            diff_t.append(float(torch.sum(torch.abs(warped_s - t))))
            if i_t > 0:
                p = abs(diff_t[-2]-diff_t[-1]) * 100 / diff_t[-2]
                if p > 2:
                    break
        if l != 1:
            ux = ux.data.cpu().numpy()
            h_, w_ = ux.shape
            ux = torch.from_numpy(resize(ux, [h_*2, w_*2], 2)).cuda()
            uy = uy.data.cpu().numpy()
            uy = torch.from_numpy(resize(uy, [h_*2, w_*2], 2)).cuda()



    # test = tdct.idct_2d(tdct.dct_2d(source))
    # diff = source - test

    b = time.time()
    print('cost time:', b-a)
    ux = ux.unsqueeze(0)
    uy = uy.unsqueeze(0)

    source = torch.from_numpy(source).cuda()
    target = torch.from_numpy(target).cuda()

    target = target.unsqueeze(0).unsqueeze(0)
    source = source.unsqueeze(0).unsqueeze(0)
    warp_img = warp(source.double(), ux, uy)
    warp_img = warp_img.data.cpu().numpy().squeeze()
    source = source.data.cpu().numpy().squeeze()
    target = target.data.cpu().numpy().squeeze()
    warp_img = rescale_intensity(warp_img) * 255
    # es = rescale_intensity(es) * 255

    ux =  ux.data.cpu().numpy()
    uy = uy.data.cpu().numpy()
    flow = np.stack([ux,uy]).squeeze()
    flow = flow_to_hsv(flow)
    a = np.concatenate((source*255, target*255, warp_img), 1)
    # a = cv2.merge([a,a,a])
    # a = np.concatenate((a, flow), 1)


    cv2.imwrite('test.png', a)