class HparamsBase(dict):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value

class HparamsVQGAN(HparamsBase):
    def __init__(self, dataset):
        super().__init__(dataset)
        # defaults that are same for all datasets
        self.base_lr = 4.5e-6
        self.beta = 0.25
        self.diff_aug = False
        self.gumbel_kl_weight = 1e-8
        self.gumbel_straight_through = False
        self.quantizer = 'nearest'
        self.log_dir = 'MNIST_test5'

        self.ema_beta = 0.995
        self.ema = False

        #    training args
        self.train_steps = 20000
        self.lr = 4.5e-6
        #   logging args
        self.steps_per_checkpoint = 5000
        self.steps_per_display_output = 500
        self.steps_per_eval = 0
        self.steps_per_log = 10
        self.steps_per_save_output = 500
        self.visdom_port = 8097
        self.load_step = self.train_steps - self.steps_per_checkpoint
        self.load_dir = self.log_dir

        if self.dataset == 'MNIST':
            self.attn_resolutions = [8]
            self.batch_size = 10
            self.ch_mult = [1, 2, 2, 4, 4]
            self.codebook_size = 32
            self.disc_layers = 3
            self.disc_weight_max = 1
            self.disc_start_step = 3000
            self.emb_dim = 32
            self.img_size = 32
            self.latent_shape = [1, 4, 4]
            self.n_channels = 1
            self.ndf = 64
            self.nf = 64
            self.perceptual_weight = 1.0
            self.res_blocks = 2
        elif self.dataset == 'maps':
            self.attn_resolutions = [5]
            self.batch_size = 32
            self.ch_mult = [1, 4]
            self.codebook_size = 256
            self.disc_layers = 1
            self.disc_weight_max = 1
            self.disc_start_step = 5000
            self.emb_dim = 64
            self.img_size = 10
            self.latent_shape = [1, 5, 5]
            self.n_channels = 16
            self.ndf = 64
            self.nf = 64
            self.perceptual_weight = 1.0
            self.res_blocks = 2
        elif self.dataset == 'churches' or self.dataset == "bedrooms":
            self.attn_resolutions = [16]
            self.batch_size = 3
            self.ch_mult = [1, 1, 2, 2, 4]
            self.codebook_size = 1024
            self.disc_layers = 3
            self.disc_weight_max = 1
            self.disc_start_step = 30001
            self.emb_dim = 256
            self.img_size = 256
            self.latent_shape = [1, 16, 16]
            self.n_channels = 3
            self.ndf = 64
            self.nf = 128
            self.perceptual_weight = 1.0
            self.res_blocks = 2

        elif self.dataset == 'ffhq':
            self.attn_resolutions = [16]
            self.batch_size = 3
            self.ch_mult = [1, 1, 2, 2, 4]
            self.codebook_size = 1024
            self.disc_layers = 3
            self.disc_weight_max = 1
            self.disc_start_step = 30001
            self.emb_dim = 256
            self.img_size = 256
            self.latent_shape = [1, 16, 16]
            self.n_channels = 3
            self.ndf = 64
            self.nf = 128
            self.perceptual_weight = 1.0
            self.res_blocks = 2

        elif self.dataset == 'minecraft':
            self.attn_resolutions = [6]  # Attention at 4x4x4 resolution
            self.batch_size = 8
            self.ch_mult = [1, 2, 2, 4]  # Progressive downsampling to 2x2x2
            self.codebook_size = 512  # Increased due to 3D complexity
            self.disc_layers = 3
            self.disc_weight_max = 1
            self.disc_start_step = 20000
            self.emb_dim = 256  # Increased embedding dimension
            self.img_size = 24  # 16x16x16 chunks
            self.latent_shape = [1, 6, 6, 6]  # 3D latent space
            self.n_channels = 39  # Number of block types
            self.ndf = 64
            self.nf = 64
            self.perceptual_weight = 1.0
            self.res_blocks = 2

        else:
            raise KeyError(f'Defaults not defined for VQGAN model on dataset: {self.dataset}')