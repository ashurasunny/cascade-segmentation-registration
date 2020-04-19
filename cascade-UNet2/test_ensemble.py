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


def MSE(input, target):
    return ((input-target)**2).mean()


def foward_network(a, m, model1, model2):
    a = np.expand_dims(a, axis=1)
    a = torch.from_numpy(a).to(model1.device)


    model1.set_input({'A':a, 'M':a})
    fake_M, fake_B_1, fake_B_2 = model1.test()  # run inference
    # output = softmax(output, dim=1)
    # output = torch.argmax(output, dim=1)
    fake_M = fake_M.data.to('cpu').numpy()
    fake_B_1 = fake_B_1.data.to('cpu').numpy()
    fake_B_2 = fake_B_2.data.to('cpu').numpy()

    model2.set_input({'A': a, 'M': a})
    fake_M_2, fake_B_1_2, fake_B_2_2 = model2.test()  # run inference
    # output = softmax(output, dim=1)
    # output = torch.argmax(output, dim=1)
    fake_M_2 = fake_M_2.data.to('cpu').numpy()
    fake_B_1_2 = fake_B_1_2.data.to('cpu').numpy()
    fake_B_2_2 = fake_B_2_2.data.to('cpu').numpy()

    fake_M = fake_M+fake_M_2
    fake_B_1 = fake_B_1+fake_B_1_2
    fake_B_2 = fake_B_2+fake_B_2_2

    return np.concatenate([fake_M, fake_B_1, fake_B_2], axis=1)


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

    iter1 = 187
    name1 = 'unetcascade'
    iter2 = 197
    name2 = 'unetcascade'
    # iter2 = 86
    # name2 = 'model1_1_g4g8_Stage40_40_120_Loss500_5'

    csv_name = './' + opt.name + '.csv'
    # if not os._exists('./submit'):
    #     os.mkdir('./submit')
    with open(csv_name, 'w+') as f:
        writer = csv.writer(f)

        for iter in range(1,2):
            opt.load_iter = iter1
            opt.name = name1
            model1 = create_model(opt)  # create a model given opt.model and other options
            model1.setup(opt)  # regular setup: load and print networks; create schedulers

            opt.load_iter = iter2
            opt.name = name2
            model2 = create_model(opt)  # create a model given opt.model and other options
            model2.setup(opt)  # regular setup: load and print networks; create schedulers

            instance_dice = {0: 0, 1: 0, 2: 0, 3: 0}
            instance_HD = {0: 0, 1: 0, 2: 0, 3: 0}
            instance_MSE =0
            if opt.eval:
                model1.eval()
                model2.eval()
            for patient in data_list:
                # if not '080' in patient:
                #     continue
                print(patient)
                patient_root = os.path.join(opt.dataroot, patient)

                for phase in ['ED', 'ES']:
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


                    output1 = foward_network(crop1, crop1, model1, model2)
                    output2 = foward_network(crop2, crop1, model1, model2)
                    output3 = foward_network(crop3, crop1, model1, model2)
                    output4 = foward_network(crop4, crop1, model1, model2)
                    output5 = foward_network(crop5, crop1, model1, model2)

                    # full_output = torch.argmax(torch.from_numpy(output1[:, 1:5, :, :]), dim=1)
                    # full_output = full_output.data.numpy().squeeze().astype(np.uint8)
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

                    fake_B_1 = torch.argmax(torch.from_numpy(fake_B_1), dim=1)
                    fake_B_1 = fake_B_1.data.numpy().squeeze().astype(np.uint8)
                    fake_B_2 = torch.argmax(torch.from_numpy(fake_B_2), dim=1)
                    fake_B_2 = fake_B_2.data.numpy().squeeze().astype(np.uint8)

                    tmp = convert_to_one_hot(fake_B_2)
                    vals = np.unique(fake_B_2)
                    results = []
                    for i in range(len(tmp)):
                        results.append(resize_image(tmp[i].astype(float), spacing_target, spacing, 1)[None])
                    fake_B_2 = vals[np.vstack(results).argmax(0)]

                    fake_B_2 = postprocess_prediction(fake_B_2)

                    out = sitk.GetImageFromArray(fake_B_2)

                    sitk.WriteImage(out, os.path.join('./submit',
                                                      patient + '_' + phase + '.nii.gz'))
