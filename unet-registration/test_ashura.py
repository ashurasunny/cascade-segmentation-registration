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


def MSE(input, target):
    return ((input-target)**2).mean()


def foward_network(a, b, model):
    a = np.expand_dims(a, axis=1)
    a = torch.from_numpy(a).to(model.device)
    b = np.expand_dims(b, axis=1)
    b = torch.from_numpy(b).to(model.device)

    model.set_input({'ED':a, 'ES':b})
    fake_ED_M, fake_ES_M, fake_ED_2, fake_ES_2, flow_2, warp_img, warped_mask = model.test()  # run inference
    # output = softmax(output, dim=1)
    # output = torch.argmax(output, dim=1)
    fake_ED_M = fake_ED_M.data.to('cpu').numpy()
    fake_ES_M = fake_ES_M.data.to('cpu').numpy()
    fake_ED_2 = fake_ED_2.data.to('cpu').numpy()
    fake_ES_2 = fake_ES_2.data.to('cpu').numpy()
    flow_2 = flow_2.data.to('cpu').numpy()
    warp_img = warp_img.data.to('cpu').numpy()
    warped_mask = warped_mask.data.to('cpu').numpy()
    return np.concatenate([fake_ED_M, fake_ES_M, fake_ED_2, fake_ES_2, flow_2, warp_img, warped_mask], axis=1)


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

    csv_name = './' + opt.name + '.csv'





    with open(csv_name, 'w+') as f:
        writer = csv.writer(f)

        for iter in range(200,201,5):
            opt.load_iter = iter
            model = create_model(opt)  # create a model given opt.model and other options
            model.setup(opt)  # regular setup: load and print networks; create schedulers
            instance_dice_ED = {0: 0, 1: 0, 2: 0, 3: 0}
            instance_dice_ES = {0: 0, 1: 0, 2: 0, 3: 0}
            instance_dice2 = {0: 0, 1: 0, 2: 0, 3: 0}
            instance_HD_ED = {0: 0, 1: 0, 2: 0, 3: 0}
            instance_HD_ES = {0: 0, 1: 0, 2: 0, 3: 0}
            instance_MSE = 0
            instance_diceM = 0
            if True:
                model.eval()
            for patient in data_list:
                # if not '080' in patient:
                #     continue
                print(patient)
                patient_root = os.path.join(opt.dataroot, patient)

                for phase in ['ED']:
                    # read iamge
                    itk_image = sitk.ReadImage(os.path.join(patient_root, phase + '.nii.gz'))
                    spacing = np.array(itk_image.GetSpacing())[[2, 1, 0]]
                    image = sitk.GetArrayFromImage(itk_image).astype(float)

                    image_backup = image

                    # resample image to target spacing, keep z spacing, do not modify
                    spacing_target = list(spacing_target)
                    spacing_target[0] = spacing[0]
                    image = resize_image(image, spacing, spacing_target).astype(np.float32)
                    image = rescale_intensity(image, (0.5, 99.5))

                    src = image
                    gt = sitk.ReadImage(os.path.join(patient_root, phase + '_gt.nii.gz'))
                    gt = sitk.GetArrayFromImage(gt)

                    # if True:
                    #     gt = resize_image(gt.astype(np.float32), spacing, spacing_target).astype(np.float32)
                    #     cv2.imwrite('gt_fake.png', gt[5] * 85)

                    gt_m = gt.copy().astype(np.float32)
                    gt_m[gt_m > 0] = 1
                    tmp = convert_to_one_hot(gt_m)
                    vals = np.unique(gt_m)
                    results = []

                    for i in range(len(tmp)):
                        results.append(resize_image(tmp[i].astype(float), spacing, spacing_target, 1)[None])
                    gt_m = vals[np.vstack(results).argmax(0)]


                    real_m = src * gt_m
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

                        #pad m as well
                        res = np.zeros([c, window_size, w], dtype=src.dtype)
                        res[:, start:start + h, :] = real_m
                        real_m = res

                        #update parameters
                        h = window_size
                        starth = start

                    startw = 0
                    if w < window_size:
                        # print('img width is too small!!!!!!!!')
                        res = np.zeros([c, h, window_size], dtype=src.dtype)
                        start = (window_size - w) // 2
                        res[:, :, start:start + w] = src
                        src = res

                        #pad m as well
                        res = np.zeros([c, h, window_size], dtype=src.dtype)
                        res[:, :, start:start + w] = real_m
                        real_m = res

                        #update parameters
                        startw = start
                        w = window_size

                    c_startw = (w - window_size) // 2
                    c_starth = (h - window_size) // 2

                    # 5 crop,
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

                    output1 = foward_network(crop1, crop1_m, model)
                    output2 = foward_network(crop2, crop2_m, model)
                    output3 = foward_network(crop3, crop3_m, model)
                    output4 = foward_network(crop4, crop4_m, model)
                    output5 = foward_network(crop5, crop5_m, model)

                    full_output = torch.argmax(torch.from_numpy(output1[:, 1:5, :, :]), dim=1)
                    full_output = full_output.data.numpy().squeeze().astype(np.uint8)
                    # cv2.imwrite('./test.png', full_output[5] * 85)



                    n_, h_, w_ = src.shape
                    probshape = [n_, 9, h_, w_]
                    full_output = np.zeros(probshape)

                    full_output[:, :, :window_size, :window_size] += output1
                    full_output[:, :, -window_size:, :window_size] += output2
                    full_output[:, :, :window_size, -window_size:] += output3
                    full_output[:, :, -window_size:, -window_size:] += output4
                    full_output[:, :, c_starth:c_starth + window_size, c_startw:c_startw + window_size] += output5

                    full_index = np.zeros(probshape)
                    full_index[:, :, :window_size, :window_size] += 1
                    full_index[:, :, -window_size:, :window_size] += 1
                    full_index[:, :, :window_size, -window_size:] += 1
                    full_index[:, :, -window_size:, -window_size:] += 1
                    full_index[:, :, c_starth:c_starth + window_size, c_startw:c_startw + window_size] += 1
                    full_index[full_index==0] = 1
                    full_output /= full_index

                    c, h, w = src_backup.shape
                    full_output = full_output[:, :, starth:starth + h, startw:startw + w]
                    fake_M = full_output[:,0,:,:].astype(np.float32)
                    fake_B_1 = full_output[:, 1:5, :, :].astype(np.float32)
                    fake_B_2 = full_output[:, 5:9, :, :].astype(np.float32)

                    print('realA: {}_{}, real_M: {}_{}, fakeM: {}_{}, realB: {}_{}, fakeB1:{}_{}, fakeB2:{}_{}'.format(src.min(), src.max(), real_m.min(), real_m.max(), fake_M.min(),
                                                                                                                       fake_M.max(), gt.min(), gt.max(), fake_B_1.min(), fake_B_1.max(),
                                                                                                                       fake_B_2.min(), fake_B_2.max()))
                    fake_B_1 = torch.argmax(torch.from_numpy(fake_B_1), dim=1)
                    fake_B_1 = fake_B_1.data.numpy().squeeze().astype(np.uint8)
                    fake_B_2 = torch.argmax(torch.from_numpy(fake_B_2), dim=1)
                    fake_B_2 = fake_B_2.data.numpy().squeeze().astype(np.uint8)


                    fake_M = resize_image(fake_M, spacing_target, spacing, 3).squeeze()
                    # fake_B_1 = resize_image(fake_B_1, spacing_target, spacing, 1).squeeze()
                    # fake_B_2 = resize_image(fake_B_2, spacing_target, spacing, 1).squeeze()
                    real_m_backup = resize_image(real_m_backup, spacing_target, spacing, 3).squeeze()

                    tmp = convert_to_one_hot(fake_B_1)
                    vals = np.unique(fake_B_1)
                    results = []
                    for i in range(len(tmp)):
                        results.append(resize_image(tmp[i].astype(float), spacing_target, spacing, 1)[None])
                    fake_B_1 = vals[np.vstack(results).argmax(0)]

                    tmp = convert_to_one_hot(fake_B_2)
                    vals = np.unique(fake_B_2)
                    results = []
                    for i in range(len(tmp)):
                        results.append(resize_image(tmp[i].astype(float), spacing_target, spacing, 1)[None])
                    fake_B_2 = vals[np.vstack(results).argmax(0)]
                    # fake_B_2 = postprocess_prediction(fake_B_2)

                    Z = full_output.shape[0]
                    image_backup = image_backup.astype(np.float32)
                    image_backup -= image_backup.min()
                    image_backup = image_backup * 255.0 / image_backup.max()

                    # ins++ce_diceM += np_categorical_dice_M(real_m_backup, fake_M)

                    real_m_backup = real_m_backup.astype(np.float32)
                    real_m_backup -= real_m_backup.min()
                    real_m_backup = real_m_backup * 255.0 / real_m_backup.max()

                    fake_M = fake_M.astype(np.float32)
                    fake_M -= fake_M.min()
                    fake_M = fake_M * 255.0 / fake_M.max()


                    for z in range(Z):
                        input_slice = image_backup[z, :, :]
                        output1_slice = fake_B_1[z, :, :] * 85
                        output2_slice = fake_B_2[z, :, :] * 85
                        gt_slice = gt[z, :, :] * 85
                        gtm_slice = real_m_backup[z, :, :]
                        m_slice = fake_M[z, :, :]
                        merge = np.concatenate([input_slice, m_slice, gtm_slice, output1_slice, output2_slice, gt_slice], axis=1)
                        cv2.imwrite(os.path.join(output_dir, patient + '_' + phase + '_' + str(z) + '.png'), merge)

                    for i in range(1, 4):
                        dice = np_categorical_dice(fake_B_2, gt, i)
                        dice2 = np_categorical_dice(fake_B_1, gt, i)
                        # dice_temp.append(dice)
                        if phase == 'ED':
                            instance_dice_ED[i] += round(dice, 4)
                        elif phase == 'ES':
                            instance_dice_ES[i] += round(dice, 4)
                        else:
                            print('Error!!!!!!!!!!')
                        instance_dice2[i] += round(dice2, 4)
                        try:
                            A = sitk.GetImageFromArray(fake_B_2)
                            B = sitk.GetImageFromArray(gt)
                            hausdorffcomputer.Execute(A == i, B == i)
                            HD = hausdorffcomputer.GetHausdorffDistance()
                            if phase == 'ED':
                                instance_HD_ED[i] += HD
                            elif phase == 'ES':
                                instance_HD_ES[i] += HD

                            print(patient, phase, i, dice, HD)
                        except:
                            print("Hausdorff Eroor!   ", patient, '  ', phase)

            for i in range(1, 4):
                instance_dice_ES[i] /= (1 * len(data_list))
                instance_dice_ED[i] /= (1 * len(data_list))
                instance_dice2[i] /= (2 * len(data_list))
                instance_HD_ES[i] /= (1 * len(data_list))
                instance_HD_ED[i] /= (1 * len(data_list))
            # instance_MSE /= (2 * len(data_list))
            # instance_diceM /= (2 * len(data_list))
            print('iteration{}:'.format(iter), instance_dice_ED,
                  (instance_dice_ED[1] + instance_dice_ED[2] + instance_dice_ED[3]) / 3)
            print('iteration{}:'.format(iter), instance_dice_ES,
                  (instance_dice_ES[1] + instance_dice_ES[2] + instance_dice_ES[3]) / 3)
            # print(instance_diceM)
            print(instance_HD_ED, (instance_HD_ED[1] + instance_HD_ED[2] + instance_HD_ED[3]) / 3)
            print(instance_HD_ES, (instance_HD_ES[1] + instance_HD_ES[2] + instance_HD_ES[3]) / 3)
            # HD_mean = (instance_HD[1] + instance_HD[2] + instance_HD[3]) / 3
            # dice_mean = (instance_dice[1] + instance_dice[2] + instance_dice[3]) / 3
            # dice_mean2 = (instance_dice2[1] + instance_dice2[2] + instance_dice2[3]) / 3
            temp_list = []
            temp_list.append(iter)
            temp_mean = 0
            for i in range(1, 4):
                temp_list.append(instance_dice_ED[i])
                temp_mean += instance_dice_ED[i]
            temp_mean /= 3
            temp_list.append(temp_mean)
            temp_mean = 0
            for i in range(1, 4):
                temp_list.append(instance_dice_ES[i])
                temp_mean += instance_dice_ES[i]
            temp_mean /= 3
            temp_list.append(temp_mean)
            temp_mean = 0
            for i in range(1, 4):
                temp_list.append(instance_HD_ED[i])
                temp_mean += instance_HD_ED[i]
            temp_mean /= 3
            temp_list.append(temp_mean)
            temp_mean = 0
            for i in range(1, 4):
                temp_list.append(instance_HD_ES[i])
                temp_mean += instance_HD_ES[i]
            temp_mean /= 3
            temp_list.append(temp_mean)


            writer.writerow(temp_list)
