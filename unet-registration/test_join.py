"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from utils import *
import csv
from models.pix2pix_model import warp


def np_categorical_dice(pred, truth, k):
    """ Dice overlap metric for label k """
    A = (pred == k).astype(np.float32)
    B = (truth == k).astype(np.float32)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B))


def np_categorical_dice_M(pred, truth):
    """ Dice overlap metric for label k """
    A = pred.astype(np.float32)
    B = truth.astype(np.float32)
    return (2 * np.sum(A * B)+0.000001) / ((np.sum(A*A) + np.sum(B*B))+0.000001)

def norm_255(image_backup):
    image_backup = image_backup.astype(np.float32)
    image_backup -= image_backup.min()
    image_backup = image_backup * 255.0 / image_backup.max()
    return image_backup


def MSE(input, target):
    return ((input-target)**2).mean()


def foward_network(a, b, a_, b_, ed_gt, model):
    h, w = a.shape[1:3]

    a = np.expand_dims(a, axis=1)
    gridimg = np.zeros(a.shape).astype(np.float32)
    s = 5
    sh = h // s
    sw = w // s
    for i in range(sh):
        gridimg[:, :,i * s, :] = 255
    for i in range(sw):
        gridimg[:, :, :, i * s] = 255
    gridimg = torch.from_numpy((gridimg)).to(model.device)
    a = torch.from_numpy(a).to(model.device)
    b = np.expand_dims(b, axis=1)
    b = torch.from_numpy(b).to(model.device)
    a_ = np.expand_dims(a_, axis=1)
    a_ = torch.from_numpy(a_).to(model.device)
    b_ = np.expand_dims(b_, axis=1)
    b_ = torch.from_numpy(b_).to(model.device)
    ed_gt = np.expand_dims(ed_gt, axis=1)
    ed_gt = torch.from_numpy(ed_gt)

    model.set_input({'ED':a, 'ES':b, 'ED_gt':ed_gt, 'ED_M':a_, 'ES_M':b_})
    flow_2, warp_img, warped_mask = model.test()  # run inference

    gridimg = warp(gridimg, flow_2[:, 0, :, :], flow_2[:, 1:, :], interp='bilinear').to('cpu').numpy()
    # cv2.imwrite('test.png', gridimg[5, :,:, :].squeeze())

    flow_2 = flow_2.data.to('cpu').numpy()
    warp_img = warp_img.data.to('cpu').numpy()
    warped_mask = warped_mask.data.to('cpu').numpy()
    return np.concatenate([flow_2, warp_img, warped_mask, gridimg], axis=1)


def flow_to_hsv(opt_flow, max_mag=0.2, white_bg=True):
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
    mag = 1.3 * mag / max_mag
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

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    output_dir = './output'
    spacing_target = (10, 1.25, 1.25)
    window_size = 256
    stride = 128
    batch_size = 1
    data_list = os.listdir(opt.dataroot)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()

    test_list = []
    for patient in data_list:
        # if  '020' in patient or '035' in patient or '037' in patient or '075' in patient or '080' in patient:
        # if not ('035' in patient or '034' in patient):
        #     print('nnnnn', patient)
        #     continue
        print(patient)
        patient_root = os.path.join(opt.dataroot, patient)
        temp_list = []
        for phase in ['ED', 'ES']:
            # read iamge
            itk_image = sitk.ReadImage(os.path.join(patient_root, phase + '.nii.gz'))
            spacing = np.array(itk_image.GetSpacing())[[2, 1, 0]]
            image = sitk.GetArrayFromImage(itk_image).astype(float)
            image = rescale_intensity_slice(image, (0.5, 99.5))
            image_backup = image

            # resample image to target spacing, keep z spacing, do not modify
            spacing_target = list(spacing_target)
            spacing_target[0] = spacing[0]
            image = resize_image(image, spacing, spacing_target).astype(np.float32)


            src = image
            gt = sitk.ReadImage(os.path.join(patient_root, phase + '_gt.nii.gz'))
            gt = sitk.GetArrayFromImage(gt)

            gt_m = gt.copy().astype(np.float32)
            # gt_m[gt_m > 0] = 1

            tmp = convert_to_one_hot(gt_m)
            vals = np.unique(gt_m)
            results = []
            for i in range(len(tmp)):
                results.append(resize_image(tmp[i].astype(float), spacing, spacing_target, 1)[None])
            gt_m = vals[np.vstack(results).argmax(0)]


            gt_mm = gt.copy().astype(np.float32)
            gt_mm[gt_mm>0] = 1

            tmp = convert_to_one_hot(gt_mm)
            vals = np.unique(gt_mm)
            results = []
            for i in range(len(tmp)):
                results.append(resize_image(tmp[i].astype(float), spacing, spacing_target, 1)[None])
            gt_mm = vals[np.vstack(results).argmax(0)]
            real_m = src * gt_mm
            real_m_backup = real_m.copy()

            c, h, w = src.shape
            src_backup = src

            # crop image, if image size < crop size, pad at first
            starth = 0
            if h < window_size:
                # print('img height is too small!!!!!!!!')
                res = np.zeros([c, window_size, w], dtype=src.dtype)
                start = (window_size - h) // 2
                res[:, start:start + h, :] = src
                src = res

                # pad m as well
                res = np.zeros([c, window_size, w], dtype=src.dtype)
                res[:, start:start + h, :] = real_m
                real_m = res

                res = np.zeros([c, window_size, w], dtype=src.dtype)
                res[:, start:start + h, :] = gt_m
                gt_m = res

                # update parameters
                h = window_size
                starth = start

            startw = 0
            if w < window_size:
                # print('img width is too small!!!!!!!!')
                res = np.zeros([c, h, window_size], dtype=src.dtype)
                start = (window_size - w) // 2
                res[:, :, start:start + w] = src
                src = res

                # pad m as well
                res = np.zeros([c, h, window_size], dtype=src.dtype)
                res[:, :, start:start + w] = real_m
                real_m = res

                res = np.zeros([c, h, window_size], dtype=src.dtype)
                res[:, :, start:start + w] = gt_m
                gt_m = res

                # update parameters
                w = window_size
                startw = start

            c_startw = (w - window_size) // 2
            c_starth = (h - window_size) // 2

            src = src[4:5, :, :]
            real_m = real_m[4:5, :,:]
            gt_m = gt_m[4:5, : ,:]
            image_backup = image_backup[4:5, :, :]
            gt = gt[4:5, :, :]

            crop1 = src[:, :window_size, :window_size]
            crop2 = src[:, -window_size:, :window_size]
            crop3 = src[:, :window_size:, -window_size:]
            crop4 = src[:, -window_size:, -window_size:]
            crop5 = src[:, c_starth:c_starth + window_size, c_startw:c_startw + window_size]

            crop1_m = real_m[:, :window_size, :window_size]
            crop2_m = real_m[:, -window_size:, :window_size]
            crop3_m = real_m[:, :window_size:, -window_size:]
            crop4_m = real_m[:, -window_size:, -window_size:]
            crop5_m = real_m[:, c_starth:c_starth + window_size, c_startw:c_startw + window_size]

            crop1_gt = gt_m[:, :window_size, :window_size]
            crop2_gt = gt_m[:, -window_size:, :window_size]
            crop3_gt = gt_m[:, :window_size:, -window_size:]
            crop4_gt = gt_m[:, -window_size:, -window_size:]
            crop5_gt = gt_m[:, c_starth:c_starth + window_size, c_startw:c_startw + window_size]

            temp_list.append([[crop1, crop2, crop3, crop4, crop5], [crop1_m, crop2_m, crop3_m, crop4_m, crop5_m],
                              [spacing, c_starth, c_startw, starth, startw, src.shape,src_backup.shape, spacing_target], [patient, phase], [image_backup, gt],
                              [crop1_gt, crop2_gt, crop3_gt, crop4_gt, crop5_gt]])

        test_list.append(temp_list)


    csv_name = './' + opt.name + '.csv'
    with open(csv_name, 'w+') as f:
        writer = csv.writer(f)

        for iter in range(5,201 ,5):
            opt.load_iter = iter
            model = create_model(opt)  # create a model given opt.model and other options
            model.setup(opt)  # regular setup: load and print networks; create schedulers
            instance_dice_ED = {0: 0, 1: 0, 2: 0, 3: 0}
            instance_dice_ES = {0: 0, 1: 0, 2: 0, 3: 0}
            instance_dice_wp = {0: 0, 1: 0, 2: 0, 3: 0}
            instance_HD_ED = {0: 0, 1: 0, 2: 0, 3: 0}
            instance_HD_ES = {0: 0, 1: 0, 2: 0, 3: 0}
            instance_HD_wp = {0: 0, 1: 0, 2: 0, 3: 0}
            instance_MSE = 0
            instance_diceM = 0
            if True:
                model.eval()
            for index in range(len(test_list)):
                # if not '080' in patient:
                #     continue
                ED_imgs = test_list[index][0]
                ES_imgs = test_list[index][1]
                temp = ED_imgs
                ED_imgs = ES_imgs
                ES_imgs = temp

                spacing, c_starth, c_startw, starth, startw, src_shape, src_backup_shape, spacing_target = test_list[index][0][2]
                ED_src = ED_imgs[4][0]
                ES_src = ES_imgs[4][0]
                ED_gt = ED_imgs[4][1]
                ES_gt = ES_imgs[4][1]

                patient, phase = ED_imgs[3]
                # if '011' not in patient:
                #     continue
                print(patient, phase)




                output1 = foward_network(ED_imgs[0][0], ES_imgs[0][0], ED_imgs[1][0], ES_imgs[1][0], ED_imgs[5][0],model)
                output2 = foward_network(ED_imgs[0][1], ES_imgs[0][1], ED_imgs[1][1], ES_imgs[1][1], ED_imgs[5][1],model)
                output3 = foward_network(ED_imgs[0][2], ES_imgs[0][2], ED_imgs[1][2], ES_imgs[1][2], ED_imgs[5][2],model)
                output4 = foward_network(ED_imgs[0][3], ES_imgs[0][3], ED_imgs[1][3], ES_imgs[1][3], ED_imgs[5][3],model)
                output5 = foward_network(ED_imgs[0][4], ES_imgs[0][4], ED_imgs[1][4], ES_imgs[1][4], ED_imgs[5][4],model)

                n_, h_, w_ = src_shape
                probshape = [n_, 5, h_, w_]
                full_output = np.zeros(probshape)

                # full_output[:, :, :window_size, :window_size] += output1
                # full_output[:, :, -window_size:, :window_size] += output2
                # full_output[:, :, :window_size, -window_size:] += output3
                # full_output[:, :, -window_size:, -window_size:] += output4
                full_output[:, :, c_starth:c_starth + window_size, c_startw:c_startw + window_size] += output5

                full_index = np.zeros(probshape)
                full_index[:, :, :window_size, :window_size] += 1
                full_index[:, :, -window_size:, :window_size] += 1
                full_index[:, :, :window_size, -window_size:] += 1
                full_index[:, :, -window_size:, -window_size:] += 1
                full_index[:, :, c_starth:c_starth + window_size, c_startw:c_startw + window_size] += 1
                # full_output /= full_index

                c, h, w = src_backup_shape
                full_output = full_output[:, :, starth:starth + h, startw:startw + w]

                # np.concatenate([fake_ED_M, fake_ES_M, fake_ED_2, fake_ES_2, flow_2, warp_img, warped_mask], axis=1)


                flow_2 = full_output[:, 0:2, :, :].astype(np.float32)
                flow0 = resize_image(flow_2[:, 0, :, :], spacing_target, spacing, 3)

                flow1 = resize_image(flow_2[:, 1, :, :], spacing_target, spacing, 3)
                flow_2 = np.stack([flow0, flow1], axis=1)

                # flow_2 = flow_2.transpose([0,2,3,1])
                warped_img = full_output[:, 2, :, :].astype(np.float32)
                warped_mask = full_output[:, 3, :, :].astype(np.float32)
                warped_img = resize_image(warped_img, spacing_target, spacing, 3).squeeze()
                warped_mask = resize_image(warped_mask, spacing_target, spacing, 1).squeeze()
                warped_mask_oh = warped_mask.astype(np.uint8)
                ES_gt_oh = ES_gt.astype(np.uint8)

                for i in range(1, 4):
                    try:
                        dice = np_categorical_dice(warped_mask, ES_gt_oh, i)
                        instance_dice_wp[i] += round(dice, 4)
                        print(i,dice)
                    except:
                        print("Dice Eroor!   ", patient, '  ', phase)
                    try:
                        A = sitk.GetImageFromArray(warped_mask_oh)
                        B = sitk.GetImageFromArray(ES_gt_oh)
                        hausdorffcomputer.Execute(A == i, B == i)
                        HD = hausdorffcomputer.GetHausdorffDistance()
                        # if phase == 'ED':
                        #     instance_HD_ED[i] += HD
                        # elif phase == 'ES':
                        #     instance_HD_ES[i] += HD
                        instance_HD_wp[i] += HD

                        print(patient, phase, i, dice, HD)
                    except:
                        print("Hausdorff Eroor!   ", patient, '  ', phase)


                gridimg = full_output[:, 4, :, :].astype(np.float32)
                gridimg = resize_image(gridimg, spacing_target, spacing, 3).squeeze()


                Z = full_output.shape[0]
                ED_src = norm_255(ED_src)
                ES_src = norm_255(ES_src)
                warped_img = norm_255(warped_img)




                # for z in range(Z):
                #     ED_slice = ED_src[z, :, :]
                #     ES_slice = ES_src[z, :, :]
                #
                #
                #     ED_gt_slice = ED_gt[z, :, :] * 85
                #     ES_gt_slice = ES_gt[z, :, :] * 85
                #
                #     warped_img_slice = warped_img[z, :, :]
                #     warped_mask_slice= warped_mask[z, :, :] * 85
                #
                #     flow_slice = flow_2[z,:,:,:]
                #     flow_slice = flow_to_hsv(flow_slice)
                #
                #     merge = np.concatenate([ED_slice, ED_gt_slice, ES_slice, ES_gt_slice, warped_img_slice, warped_mask_slice, gridimg[z, :, :]],
                #                            axis=1)
                #     merge = cv2.merge([merge,merge, merge])
                #     merge = np.concatenate([merge, flow_slice], axis=1)
                #     cv2.imwrite(os.path.join(output_dir, patient + '_' + phase + '_' + str(z) + '.png'), merge)

                ED_slice = ED_src.squeeze()
                ES_slice = ES_src.squeeze()

                ED_gt_slice = ED_gt.squeeze() * 85
                ES_gt_slice = ES_gt.squeeze() * 85

                warped_img_slice = warped_img
                warped_mask_slice = warped_mask * 85




                flow_slice = flow_2.squeeze()
                print(patient, flow_slice.shape)
                flow_slice = flow_to_hsv(flow_slice)

                merge = np.concatenate(
                    [ED_slice, ED_gt_slice, ES_slice, ES_gt_slice, warped_img_slice, warped_mask_slice,
                     gridimg], axis=1)
                merge = cv2.merge([merge, merge, merge])
                merge = np.concatenate([merge, flow_slice], axis=1)
                cv2.imwrite(os.path.join(output_dir, patient + '_' + phase + '_' + '.png'), merge)



            for i in range(1, 4):
                # instance_dice_ES[i] /= (1 * len(data_list))
                # instance_dice_ED[i] /= (1 * len(data_list))
                # instance_HD_ES[i] /= (1 * len(data_list))
                # instance_HD_ED[i] /= (1 * len(data_list))
                instance_dice_wp[i] /= (1 * len(test_list))
                instance_HD_wp[i] /= (1 * len(test_list))
            # instance_MSE /= (2 * len(data_list))
            # instance_diceM /= (2 * len(data_list))
            print('iteration{}:'.format(iter), instance_dice_wp,
                  (instance_dice_wp[1] + instance_dice_wp[2] + instance_dice_wp[3]) / 3)
            print(instance_HD_wp, (instance_HD_wp[1] + instance_HD_wp[2] + instance_HD_wp[3]) / 3)
            # print(instance_HD_ES, (instance_HD_ES[1] + instance_HD_ES[2] + instance_HD_ES[3]) / 3)
            # # HD_mean = (instance_HD[1] + instance_HD[2] + instance_HD[3]) / 3
            # # dice_mean = (instance_dice[1] + instance_dice[2] + instance_dice[3]) / 3
            # # dice_mean2 = (instance_dice2[1] + instance_dice2[2] + instance_dice2[3]) / 3
            temp_list = []
            temp_list.append(iter)
            temp_mean = 0
            for i in range(1, 4):
                temp_list.append(instance_dice_wp[i])
                temp_mean += instance_dice_wp[i]
            temp_mean /= 3
            temp_list.append(temp_mean)
            # temp_mean = 0
            # for i in range(1, 4):
            #     temp_list.append(instance_dice_ES[i])
            #     temp_mean += instance_dice_ES[i]
            # temp_mean /= 3
            # temp_list.append(temp_mean)
            temp_mean = 0
            for i in range(1, 4):
                temp_list.append(instance_HD_wp[i])
                temp_mean += instance_HD_wp[i]
            temp_mean /= 3
            temp_list.append(temp_mean)
            # temp_mean = 0
            # for i in range(1, 4):
            #     temp_list.append(instance_HD_ES[i])
            #     temp_mean += instance_HD_ES[i]
            # temp_mean /= 3
            # temp_list.append(temp_mean)
            #
            #
            writer.writerow(temp_list)
