
class default_args():
    '''
    Default args we don't use
    '''
    def __init__(self):
        self.config_file = None # 'json configuration file'
        self.device_ids = [0]
        self.world_size = -1
        self.local_rank = -1
        self.dist_init = 'env://'
        self.dist_backend = 'nccl'
        self.workers = 8
        self.label_smoothing = 0
        self.mixup = None
        self.duplicates = 1
        self.chunk_batch = 1
        self.augment = False
        self.cutout = False
        self.calibrate_bn = False
        self.autoaugment = False
        self.avg_out = False
        self.print_freq = 10
        self.resume = ''
        self.absorb_bn = False
        self.seed = 123
        self.save = ''
        self.input_size = None
        self.dtype = 'float'
        self.optimizer = 'SGD'
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 0
        self.loss_scale = 1
        self.grad_clip = -1
        self.adapt_grad_norm = None
        self.tensorwatch = False
        self.start_epoch = -1

class default_td_args(default_args):
    '''
    Default args we do use in this project
    '''
    def __init__(self):
        super().__init__()
        self.dataset = 'cifar10'
        self.model = 'resnet'
        self.device = 'cuda'
        self.model_config = ''
        self.datasets_dir = '~/Datasets'
        self.results_dir = './results'

class train_args(default_td_args):
    '''
    training args
    '''
    def __init__(self):
        super().__init__()
        self.epochs = 200
        self.batch_size = 256
        self.eval_batch_size = self.batch_size
        self.pruning_perc = None
        self.pruning_policy = 'unit'
        self.evaluate = False


class eval_prune_args(default_td_args):
    '''
    pruning args
    '''
    def __init__(self):
        super().__init__()
        self.eval_path = 'td_05unit_model\\checkpoint.pth.tar'
        self.batch_size = 256
        self.pruning_perc = None
        self.pruning_percs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.pruning_policy = 'unit'
        self.gather_histograms = True