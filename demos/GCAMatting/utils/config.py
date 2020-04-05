from easydict import EasyDict

# Base default config
CONFIG = EasyDict({})
# to indicate this is a default setting, should not be changed by user
CONFIG.is_default = True
CONFIG.version = "baseline"
CONFIG.phase = "train"
# distributed training
CONFIG.dist = False
# global variables which will be assigned in the runtime
CONFIG.local_rank = 0
CONFIG.gpu = 0
CONFIG.world_size = 1

# Model config
CONFIG.model = EasyDict({})
# use pretrained checkpoint as encoder
CONFIG.model.imagenet_pretrain = True
CONFIG.model.imagenet_pretrain_path = "/home/liyaoyi/Source/python/attentionMatting/pretrain/model_best_resnet34_En_nomixup.pth"
CONFIG.model.batch_size = 16
# one-hot or class, choice: [3, 1]
CONFIG.model.trimap_channel = 3

# Model -> Architecture config
CONFIG.model.arch = EasyDict({})
# definition in networks/encoders/__init__.py and networks/encoders/__init__.py
CONFIG.model.arch.encoder = "res_shortcut_encoder_29"
CONFIG.model.arch.decoder = "res_shortcut_decoder_22"
# predefined for GAN structure
CONFIG.model.arch.discriminator = None


# Dataloader config
CONFIG.data = EasyDict({})
CONFIG.data.workers = 0
# data path for training and validation in training phase
CONFIG.data.train_fg = None
CONFIG.data.train_alpha = None
CONFIG.data.train_bg = None
CONFIG.data.test_merged = None
CONFIG.data.test_alpha = None
CONFIG.data.test_trimap = None
# feed forward image size (untested)
CONFIG.data.crop_size = 512
# composition of two foregrounds, affine transform, crop and HSV jitter
CONFIG.data.augmentation = True
CONFIG.data.random_interp = False


# Training config
CONFIG.train = EasyDict({})
CONFIG.train.total_step = 100000
CONFIG.train.warmup_step = 5000
CONFIG.train.val_step = 1000
# basic learning rate of optimizer
CONFIG.train.G_lr = 1e-3
# beta1 and beta2 for Adam
CONFIG.train.beta1 = 0.5
CONFIG.train.beta2 = 0.999
# weight of different losses
CONFIG.train.rec_weight = 1
CONFIG.train.comp_weight = 0
CONFIG.train.gabor_weight = 0
CONFIG.train.grad_weight = 0
CONFIG.train.smooth_l1_weight = 0
# clip large gradient
CONFIG.train.clip_grad = True
# resume the training (checkpoint file name)
CONFIG.train.resume_checkpoint = None
# reset the learning rate (this option will reset the optimizer and learning rate scheduler and ignore warmup)
CONFIG.train.reset_lr = False


# Testing config
CONFIG.test = EasyDict({})
# data path for evaluation
CONFIG.test.merged = None
CONFIG.test.alpha = None
CONFIG.test.trimap = None
# test image scale to evaluate, "origin" or "resize" or "crop"
CONFIG.test.scale = "origin"
# test on CPU (unimplemented)
CONFIG.test.cpu = False
# path to save alpha estimation. "None" for inference without saving.
CONFIG.test.alpha_path = None
CONFIG.test.batch_size = 1
# "best_model" or "latest_model" or other base name of the pth file.
CONFIG.test.checkpoint = "best_model"
# fast evaluation, only calculate SAD and MSE
CONFIG.test.fast_eval = True


# Logging config
CONFIG.log = EasyDict({})
CONFIG.log.tensorboard_path = "./logs/tensorboard"
CONFIG.log.tensorboard_step = 100
# save less images to save disk space
CONFIG.log.tensorboard_image_step = 500
CONFIG.log.logging_path = "./logs/stdout"
CONFIG.log.logging_step = 10
CONFIG.log.logging_level = "DEBUG"
CONFIG.log.checkpoint_path = "./checkpoints"
CONFIG.log.checkpoint_step = 10000


def load_config(custom_config, default_config=CONFIG, prefix="CONFIG"):
    """
    This function will recursively overwrite the default config by a custom config
    :param default_config:
    :param custom_config: parsed from config/config.toml
    :param prefix: prefix for config key
    :return: None
    """
    if "is_default" in default_config:
        default_config.is_default = False

    for key in custom_config.keys():
        full_key = ".".join([prefix, key])
        if key not in default_config:
            raise NotImplementedError("Unknown config key: {}".format(full_key))
        elif isinstance(custom_config[key], dict):
            if isinstance(default_config[key], dict):
                load_config(default_config=default_config[key],
                            custom_config=custom_config[key],
                            prefix=full_key)
            else:
                raise ValueError("{}: Expected {}, got dict instead.".format(full_key, type(custom_config[key])))
        else:
            if isinstance(default_config[key], dict):
                raise ValueError("{}: Expected dict, got {} instead.".format(full_key, type(custom_config[key])))
            else:
                default_config[key] = custom_config[key]


if __name__ == "__main__":
    import toml
    from pprint import pprint

    pprint(CONFIG)
    with open("../config/gca-dist.toml") as f:
        custom_config = EasyDict(toml.load(f))
    load_config(custom_config=custom_config)
    pprint(CONFIG)


