import os
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models import backbone
import utils

class GANModel(object):
    def __init__(self, params, dist_model=False):

        netD_params = params['discriminator']

        # define model
        self.model = backbone.__dict__[params['backbone_arch']](**params['backbone_param'])
        utils.init_weights(self.model, init_type='xavier')
        self.model.cuda()
        if dist_model:
            self.model = utils.DistModule(self.model)
            self.world_size = dist.get_world_size()
        else:
            self.model = backbone.FixModule(self.model)
            self.world_size = 1

        if params['optim'] == 'SGD':
            self.optim = torch.optim.SGD(
                self.model.parameters(), lr=params['lr'],
                momentum=0.9, weight_decay=params['weight_decay'])
        elif params['optim'] == 'Adam':
            self.optim = torch.optim.Adam(
                self.model.parameters(), lr=params['lr'],
                betas=(params['beta1'], 0.999))
        else:
            raise Exception("No such optimizer: {}".format(params['optim']))

        # define netD
        self.netD = backbone.__dict__[netD_params['arch']](**netD_params['arch_param'])
        self.netD.cuda()
        if dist_model:
            self.netD = utils.DistModule(self.netD)
        else:
            self.netD = backbone.FixModule(self.netD)

        if netD_params['optim'] == 'SGD':
            self.optimD = torch.optim.SGD(
                self.netD.parameters(), lr=netD_params['lr'],
                momentum=0.9, weight_decay=0.0001)
        elif netD_params['optim'] == 'Adam':
            self.optimD = torch.optim.Adam(
                self.netD.parameters(), lr=netD_params['lr'],
                betas=(netD_params['beta1'], 0.999))
        else:
            raise Exception("No such optimizer: {}".format(netD_params['optim']))

        cudnn.benchmark = True

    def set_input(self, image, target=None):
        self.image = image
        if target is not None:
            if len(target.shape) == 3:
                self.target = target.unsqueeze(1)
        else:
            self.target = None

    def eval(self, ret_loss=True):
        pass

    def step(self):
        pass

    def load_state(self, path, Iter, resume=False):
        model_path = os.path.join(path, "ckpt_iter_{}.pth.tar".format(Iter))
        discriminator_path = os.path.join(path, "D_iter_{}.pth.tar".format(Iter))

        if resume:
            utils.load_state(model_path, self.model, self.optim)
            utils.load_state(discriminator_path, self.netD, self.optimD)
        else:
            utils.load_state(model_path, self.model)
            utils.load_state(discriminator_path, self.netD)

    def save_state(self, path, Iter):
        model_path = os.path.join(path, "ckpt_iter_{}.pth.tar".format(Iter))
        discriminator_path = os.path.join(path, "D_iter_{}.pth.tar".format(Iter))

        torch.save({
            'step': Iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict()}, model_path)

        torch.save({
            'step': Iter,
            'state_dict': self.netD.state_dict(),
            'optimizer': self.optimD.state_dict()}, discriminator_path)

    def switch_to(self, phase):
        if phase == 'train':
            self.model.train()
            self.netD.train()
        else:
            self.model.eval()
            self.netD.eval()
