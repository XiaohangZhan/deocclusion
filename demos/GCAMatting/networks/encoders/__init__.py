import logging
from .resnet_enc import ResNet_D, BasicBlock
from .res_shortcut_enc import ResShortCut_D
from .res_gca_enc import ResGuidedCxtAtten


__all__ = ['res_shortcut_encoder_29', 'resnet_gca_encoder_29']


def _res_shortcut_D(block, layers, **kwargs):
    model = ResShortCut_D(block, layers, **kwargs)
    return model


def _res_gca_D(block, layers, **kwargs):
    model = ResGuidedCxtAtten(block, layers, **kwargs)
    return model


def resnet_gca_encoder_29(**kwargs):
    """Constructs a resnet_encoder_29 model.
    """
    return _res_gca_D(BasicBlock, [3, 4, 4, 2], **kwargs)


def res_shortcut_encoder_29(**kwargs):
    """Constructs a resnet_encoder_25 model.
    """
    return _res_shortcut_D(BasicBlock, [3, 4, 4, 2], **kwargs)


if __name__ == "__main__":
    import torch
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    resnet_encoder = res_shortcut_encoder_29()
    x = torch.randn(4,6,512,512)
    z = resnet_encoder(x)
    print(z[0].shape)
