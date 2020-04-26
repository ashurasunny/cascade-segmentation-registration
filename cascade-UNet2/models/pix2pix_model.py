import torch
from .base_model import BaseModel
from . import networks
from models.loss import MulticlassDiceLoss, DiceLoss_M
import torch.nn.functional as F


def warp(source, offset_w, offset_h, interp='bilinear'):
    # generate standard mesh grid
    h, w = source.shape[-2:]
    grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h), torch.linspace(-1, 1, w)])

    # stop autograd from calculating gradients on standard grid line
    grid_h = grid_h.cuda()
    grid_w = grid_w.cuda()

    # (N, 1, H, W) -> (N, H, W)
    #    print('################### offset_h: {}'.format(offset_h.shape))
    offset_h = offset_h.squeeze(1)
    offset_w = offset_w.squeeze(1)
    #    print('################### offset_h: {}'.format(offset_h.shape))

    # (h,w) + (N, h, w) add by broadcasting
    grid_h = grid_h + offset_h
    grid_w = grid_w + offset_w

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

def huber_loss(x):
    bsize, csize, height, width = x.size()
    d_x = torch.index_select(x, 3, torch.arange(1, width).cuda()) - torch.index_select(x, 3, torch.arange(width-1).cuda())
    d_y = torch.index_select(x, 2, torch.arange(1, height).cuda()) - torch.index_select(x, 2, torch.arange(height-1).cuda())
    err = torch.sum(torch.mul(d_x, d_x))/height + torch.sum(torch.mul(d_y, d_y))/width
    err /= bsize
    tv_err = torch.sqrt(0.01+err)
    return tv_err

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.
    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_Dice', type=float, default=1.0, help='weight for MSE loss')
            parser.add_argument('--lambda_MSE', type=float, default=100.0, help='weight for Dice loss')
            parser.add_argument('--lambda_huber', type=float, default=0.1, help='weight for huber loss')
            parser.add_argument('--lambda_DiceM', type=float, default=5.0, help='weight for Dice loss')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1_1', 'G_Dice_2', 'G_MSE_warpimg', 'G_Dice_warpmask', 'G_huber', 'G_MSE_flow']
        # self.loss_G_3 = self.loss_G_MSE_warpimg + self.loss_G_Dice_warpmask + self.loss_G_huber + self.loss_G_MSE_flow
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = []
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G_1', 'G_2', 'G_3']
        # define networks (both generator and discriminator)
        if opt.dataset == 'ACDC':
            #activation 1:relu 2:softmax 3:tanh
            self.netG_1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, activation=1)
            self.netG_2 = networks.define_G(2, 4, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, activation=2)
            self.netG_3 = networks.define_G(4, 2, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                            activation=3)

        if self.isTrain:
            # define loss functions
            self.criterionL1_1 = torch.nn.L1Loss()
            self.criterionDice_1 = DiceLoss_M().to(self.device)
            # self.criterionL1_2 = torch.nn.L1Loss()
            self.criterionDice_2 = MulticlassDiceLoss().to(self.device)
            self.criterionMSE = torch.nn.MSELoss()
            self.dice_weights = [opt.dice_w0, opt.dice_w1, opt.dice_w2, opt.dice_w3]
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G_1 = torch.optim.Adam(self.netG_1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_2 = torch.optim.Adam(self.netG_2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_3 = torch.optim.Adam(self.netG_3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G_1)
            self.optimizers.append(self.optimizer_G_2)
            self.optimizers.append(self.optimizer_G_3)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """
        if True:
            self.ED = input['ED'].to(self.device)
            self.ES = input['ES'].to(self.device)
            self.ED_gt = input['ED_gt'].to(self.device)
            if self.isTrain:
                self.ED_gt = input['ED_gt'].to(self.device)
                self.ES_gt = input['ES_gt'].to(self.device)
                self.ED_M = input['ED_M'].to(self.device)
                self.ES_M = input['ES_M'].to(self.device)
        else:
            self.real_A = input['A'].to(self.device)
            self.real_M = (input['M']).to(self.device)


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # print("Calling forward")
        #First Network
        self.fake_ED_M = self.netG_1(self.ED)
        self.fake_ES_M = self.netG_1(self.ES)

        #Second Network
        self.real_ED_ED_M = torch.cat((self.ED, self.ED_M), 1)
        self.fake_ED_1 = self.netG_2(self.real_ED_ED_M)
        # self.fake_B_1 = torch.nn.functional.softmax(self.fake_B_1, dim=1)
        self.fakel_ED_ED_M = torch.cat((self.ED, self.fake_ED_M), 1)
        self.fake_ED_2 = self.netG_2(self.fakel_ED_ED_M)


        self.real_ES_ES_M = torch.cat((self.ES, self.ES_M), 1)
        self.fake_ES_1 = self.netG_2(self.real_ES_ES_M)
        # self.fake_B_1 = torch.nn.functional.softmax(self.fake_B_1, dim=1)
        self.fakel_ES_ES_M = torch.cat((self.ES, self.fake_ES_M), 1)
        self.fake_ES_2 = self.netG_2(self.fakel_ES_ES_M)

        #Third Network
        self.real_ED_ES = torch.cat((self.ED, self.ED_M, self.ES, self.ES_M), 1)
        self.flow_1 = self.netG_3(self.real_ED_ES)

        self.fake_ED_ES = torch.cat((self.ED, self.fake_ED_M, self.ES, self.fake_ES_M), 1)
        self.flow_2 = self.netG_3(self.fake_ED_ES)


        # self.fake_B_2 = torch.nn.functional.softmax(self.fake_B_2, dim=1)

    def backward_G_1(self):
        # print("Calling backward_G_1")
        """Calculate GAN and L1 loss for the generator"""
        # Second, G(A) = B
        self.loss_G_L1_1 = (self.criterionL1_1(self.fake_ED_M, self.ED_M) + self.criterionL1_1(self.fake_ES_M, self.ES_M) )  * self.opt.lambda_L1
        # self.loss_G_Dice_1 = self.criterionDice_1(self.fake_M, self.real_M) * self.opt.lambda_DiceM
        # combine loss and calculate gradients
        self.loss_G_1 = self.loss_G_L1_1
        self.loss_G_1.backward(retain_graph=True)

    def backward_G_2(self):
        # print("Calling backward_G_2")
        """Calculate GAN and L1 loss for the generator"""
        # Second, G(A) = B
        self.loss_G_Dice_2 = (self.criterionDice_2(self.fake_ED_1, self.ED_gt,self.dice_weights)
                              + self.criterionDice_2(self.fake_ED_2, self.ED_gt,self.dice_weights)
                              + self.criterionDice_2(self.fake_ES_1, self.ES_gt,self.dice_weights)
                              + self.criterionDice_2(self.fake_ES_2, self.ES_gt,self.dice_weights)) * self.opt.lambda_Dice
        # combine loss and calculate gradients
        self.loss_G_2 = self.loss_G_Dice_2
        self.loss_G_2.backward(retain_graph=True)

    def backward_G_3(self):
        self.warped_img_1 = warp(self.ED, self.flow_1[:,0,:,:], self.flow_1[:,1:,:], interp='bilinear')
        self.warped_img_2 = warp(self.ED, self.flow_2[:,0,:,:], self.flow_2[:,1:,:], interp='bilinear')
        self.warped_mask_1 = warp(self.ED_gt, self.flow_1[:, 0, :, :], self.flow_1[:, 1:, :], interp='nearest')
        self.warped_mask_2 = warp(self.ED_gt, self.flow_2[:, 0, :, :], self.flow_2[:, 1:, :], interp='nearest')
        self.loss_G_MSE_warpimg = (self.criterionMSE(self.warped_img_1, self.ES) + self.criterionMSE(self.warped_img_2, self.ES)) * self.opt.lambda_MSE
        self.loss_G_Dice_warpmask = (self.criterionDice_2(self.warped_mask_1, self.ES_gt) + self.criterionDice_2(self.warped_mask_2, self.ES_gt)) * self.opt.lambda_Dice
        self.loss_G_huber = (huber_loss(self.flow_1) + huber_loss(self.flow_2)) * self.opt.lambda_huber
        self.loss_G_MSE_flow = self.criterionMSE(self.flow_1, self.flow_2) * self.opt.lambda_MSE
        self.loss_G_3 = self.loss_G_MSE_warpimg + self.loss_G_Dice_warpmask + self.loss_G_huber + self.loss_G_MSE_flow


    def optimize_parameters(self):
        self.set_requires_grad(self.netG_1, True)
        # self.fake_M = self.netG_1(self.real_A)  # compute fake images: G(A)
        self.forward()
        self.optimizer_G_1.zero_grad()  # set G's gradients to zero
        self.backward_G_1()  # calculate graidents for G
        self.optimizer_G_1.step()  # udpate G's weights

    def optimize_parameters_2(self):
        self.set_requires_grad(self.netG_2, True)
        self.set_requires_grad(self.netG_3, True)
        self.forward()
        self.set_requires_grad(self.netG_1, False)

        # update G2
        self.optimizer_G_2.zero_grad()  # set G's gradients to zero
        self.backward_G_2()  # calculate graidents for G
        self.optimizer_G_2.step()  # udpate G's weights

        self.optimizer_G_3.zero_grad()  # set G's gradients to zero
        self.backward_G_3()  # calculate graidents for G
        self.optimizer_G_3.step()  # udpate G's weights

    def optimize_parameters_3(self):
        self.set_requires_grad(self.netG_1, True)
        self.set_requires_grad(self.netG_2, True)
        self.set_requires_grad(self.netG_3, True)
        self.forward()

        # update G2, G3
        self.optimizer_G_2.zero_grad()  # set G's gradients to zero
        self.backward_G_2()  # calculate graidents for G
        self.optimizer_G_2.step()  # udpate G's weights
        self.optimizer_G_1.step()

        self.optimizer_G_3.zero_grad()  # set G's gradients to zero
        self.backward_G_3()  # calculate graidents for G
        self.optimizer_G_3.step()  # udpate G's weights
        self.optimizer_G_1.step()

        # update G1
        self.optimizer_G_1.zero_grad()  # set G's gradients to zero
        self.backward_G_1()  # calculate graidents for G
        self.optimizer_G_1.step()  # udpate G's weights

    def test(self):
        with torch.no_grad():
            # First Network
            self.fake_ED_M = self.netG_1(self.ED)
            self.fake_ES_M = self.netG_1(self.ES)

            # Second Network
            self.fakel_ED_ED_M = torch.cat((self.ED, self.fake_ED_M), 1)
            self.fake_ED_2 = self.netG_2(self.fakel_ED_ED_M)
            self.fakel_ES_ES_M = torch.cat((self.ES, self.fake_ES_M), 1)
            self.fake_ES_2 = self.netG_2(self.fakel_ES_ES_M)

            # Third Network

            self.fake_ED_ES = torch.cat((self.ED, self.fake_ED_M, self.ES, self.fake_ES_M), 1)
            self.flow_2 = self.netG_3(self.fake_ED_ES)
            self. warp_img = warp(self.ED, self.flow_2[:, 0, :, :], self.flow_2[:, 1:, :], interp='bilinear')
            self.warped_mask = warp(self.ED_gt, self.flow_2[:, 0, :, :], self.flow_2[:, 1:, :], interp='nearest')
        return self.fake_ED_M, self.fake_ES_M, self.fake_ED_2, self.fake_ES_2, self.flow_2, self.warp_img, self.warped_mask