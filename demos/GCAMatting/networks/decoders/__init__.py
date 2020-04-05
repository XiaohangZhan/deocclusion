from .resnet_dec import ResNet_D_Dec, BasicBlock
from .res_shortcut_dec import ResShortCut_D_Dec
from .res_gca_dec import ResGuidedCxtAtten_Dec


__all__ = ['res_shortcut_decoder_22', 'res_gca_decoder_22']


def _res_shortcut_D_dec(block, layers, **kwargs):
    model = ResShortCut_D_Dec(block, layers, **kwargs)
    return model


def _res_gca_D_dec(block, layers, **kwargs):
    model = ResGuidedCxtAtten_Dec(block, layers, **kwargs)
    return model


def res_shortcut_decoder_22(**kwargs):
    """Constructs a resnet_encoder_14 model.
    """
    return _res_shortcut_D_dec(BasicBlock, [2, 3, 3, 2], **kwargs)


def res_gca_decoder_22(**kwargs):
    """Constructs a resnet_encoder_14 model.
    """
    return _res_gca_D_dec(BasicBlock, [2, 3, 3, 2], **kwargs)