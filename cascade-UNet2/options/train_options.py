from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=200, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--niter', type=int, default=25, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=25, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        #  additional parameters for training strategy
        parser.add_argument('--model_update_first', type=int, default=1, help='model 1 or 2 update at first')
        parser.add_argument('--model_update_freq', type=int, default=2, help='frequency of update model by epoch')
        parser.add_argument('--G1_freq', type=int, default=1, help='G1 update frequence')
        parser.add_argument('--G2_freq', type=int, default=1, help='G2 update frequence')
        parser.add_argument('--D1_freq', type=int, default=1, help='D1 update frequence')
        parser.add_argument('--D2_freq', type=int, default=1, help='D2 update frequence')
      
        parser.add_argument('--stage1_epoch', type=int, default=1, help='epoch numbers of stage 1')
        parser.add_argument('--stage2_epoch', type=int, default=1, help='epoch numbers of stage 2')
        parser.add_argument('--stage3_epoch', type=int, default=1, help='epoch numbers of stage 3')

        parser.add_argument('--dice_w0', type=float, default=1, help='dice loss weights of label 0')
        parser.add_argument('--dice_w1', type=float, default=1, help='dice loss weights of label 1')
        parser.add_argument('--dice_w2', type=float, default=1, help='dice loss weights of label 2')
        parser.add_argument('--dice_w3', type=float, default=1, help='dice loss weights of label 3')

        self.isTrain = True
        return parser
