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
import os, csv
from options.test_options import TestOptions
from models import create_model
from utils import *
from data.cpm_dataset import make_dataset
import scipy



def np_categorical_dice(pred, truth, k):
    """ Dice overlap metric for label k """
    A = (pred == k).astype(np.float32)
    B = (truth == k).astype(np.float32)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B))


def foward_network(a, model):
    a = np.expand_dims(a, axis=0)

    a = torch.from_numpy(a).float().to(model.device)
    # m = np.expand_dims(m, axis=1)
    # m = torch.from_numpy(m).to(model.device)

    model.set_input({'A':a, 'M':a})
    fake_M, fake_B_2 = model.test()  # run inference
    # output = softmax(output, dim=1)
    # output = torch.argmax(output, dim=1)
    fake_M = fake_M.data.to('cpu').numpy()
    # fake_B_1 = fake_B_1.data.to('cpu').numpy()
    fake_B_2 = fake_B_2.data.to('cpu').numpy()
    return np.concatenate([fake_M, fake_B_2], axis=1)

def MSE(input, target):
    return ((input-target)**2).mean()


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # 
    # # create a website
    # web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    output_dir = './output'
   
    window_size = 256

    batch_size = 1
    root_image = os.path.join(opt.dataroot, 'Images')
    root_label = os.path.join(opt.dataroot, 'Labels')
    imgs = make_dataset(root_image, root_label)

    csv_name = './'+opt.name+'.csv'
    with open(csv_name, 'w+') as f:
        writer = csv.writer(f)
        for iter in range(108,109):
            opt.load_iter = iter
            model = create_model(opt)  # create a model given opt.model and other options
            model.setup(opt)  # regular setup: load and print networks; create schedulers
            instance_dice = 0
            if True:
                model.eval()
            for patient in imgs:
                # if not '_03' in patient[0]:
                #     continue
                x_path, y_path = patient
                img = cv2.imread(x_path)
                img_backupt = img.copy()
                mask = scipy.io.loadmat(y_path)['inst_map']
                mask[mask > 0] = 1
                w, h = img.shape[:2]
                window_size = 256

                img = img / 255.0

                c_startw = (w - window_size) // 2
                c_starth = (h - window_size) // 2
                img = img.transpose([2, 0, 1])

                if w == 500:

                    # 5 crop,
                    crop1 = img[::, :window_size, :window_size]
                    crop2 = img[:, -window_size:, :window_size]
                    crop3 = img[:, :window_size:, -window_size:]
                    crop4 = img[:, -window_size:, -window_size:]
                    crop5 = img[:, c_starth:c_starth + window_size, c_startw:c_startw + window_size]

                    output1 = foward_network(crop1, model)
                    output2 = foward_network(crop2, model)
                    output3 = foward_network(crop3, model)
                    output4 = foward_network(crop4, model)
                    output5 = foward_network(crop5, model)

                    n_, h_, w_ = img.shape
                    probshape = [1, 5, h_, w_]
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
                    # full_index[full_index == 0] = 1
                    full_output /= full_index
                else:
                    crop1 = img[:, :window_size, :window_size]
                    crop2 = img[:, -window_size:, :window_size]
                    crop3 = img[:, :window_size:, -window_size:]
                    crop4 = img[:, -window_size:, -window_size:]
                    crop5 = img[:, c_starth:c_starth + window_size, c_startw:c_startw + window_size]
                    crop6 = img[:, c_starth:c_starth + window_size, :window_size]
                    crop7 = img[:, -window_size:, c_startw:c_startw + window_size]
                    crop8 = img[:, :window_size:, c_startw:c_startw + window_size]
                    crop9 = img[:, c_starth:c_starth + window_size:, -window_size:]

                    output1 = foward_network(crop1, model)
                    output2 = foward_network(crop2, model)
                    output3 = foward_network(crop3, model)
                    output4 = foward_network(crop4, model)
                    output5 = foward_network(crop5, model)
                    output6 = foward_network(crop6, model)
                    output7 = foward_network(crop7, model)
                    output8 = foward_network(crop8, model)
                    output9 = foward_network(crop9, model)

                    n_, h_, w_ = img.shape
                    probshape = [1, 5, h_, w_]
                    full_output = np.zeros(probshape)

                    full_output[:, :, :window_size, :window_size] += output1
                    full_output[:, :, -window_size:, :window_size] += output2
                    full_output[:, :, :window_size, -window_size:] += output3
                    full_output[:, :, -window_size:, -window_size:] += output4
                    full_output[:, :, c_starth:c_starth + window_size, c_startw:c_startw + window_size] += output5
                    full_output[:, :, c_starth:c_starth + window_size, :window_size] += output6
                    full_output[:, :, -window_size:, c_startw:c_startw + window_size] += output7
                    full_output[:, :, :window_size:, c_startw:c_startw + window_size] += output8
                    full_output[:, :, c_starth:c_starth + window_size:, -window_size:] += output9

                    full_index = np.zeros(probshape)
                    full_index[:, :, :window_size, :window_size] += 1
                    full_index[:, :, -window_size:, :window_size] += 1
                    full_index[:, :, :window_size, -window_size:] += 1
                    full_index[:, :, -window_size:, -window_size:] += 1
                    full_index[:, :, c_starth:c_starth + window_size, c_startw:c_startw + window_size] += 1
                    full_index[:, :, c_starth:c_starth + window_size, :window_size] += 1
                    full_index[:, :, -window_size:, c_startw:c_startw + window_size] += 1
                    full_index[:, :, :window_size:, c_startw:c_startw + window_size] += 1
                    full_index[:, :, c_starth:c_starth + window_size:, -window_size:] += 1
                    full_output /= full_index

                    
                fake_M = full_output[:,0:3,:,:].squeeze().transpose([1,2,0])
                fake_B = full_output[:,3:5,:,:]
                full_output = torch.argmax(torch.from_numpy(fake_B), dim=1).data.squeeze().numpy()
                dice = np_categorical_dice(full_output, mask, 1)
                # dice_temp.append(dice)
                instance_dice += round(dice, 4)
                print(x_path, dice)

                mask = mask*255
                output = full_output * 255
                output= output.squeeze()
                mask = mask.squeeze()
                output = cv2.merge([output,output,output])
                mask = cv2.merge([mask,mask,mask])
                o = np.concatenate([img_backupt, fake_M*255, output, mask], axis=1)

                cv2.imwrite(os.path.join('./output', x_path.split('/')[-1]), o)
            instance_dice /= len(imgs)
            print(iter, instance_dice)
            print(iter, instance_dice)
            writer.writerow([iter, instance_dice])

