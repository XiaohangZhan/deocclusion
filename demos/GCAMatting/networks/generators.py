import torch
import torch.nn as nn

from   ..utils import CONFIG
from   . import encoders, decoders


class Generator(nn.Module):
    def __init__(self, encoder, decoder):

        super(Generator, self).__init__()

        if encoder not in encoders.__all__:
            raise NotImplementedError("Unknown Encoder {}".format(encoder))
        self.encoder = encoders.__dict__[encoder]()

        if decoder not in decoders.__all__:
            raise NotImplementedError("Unknown Decoder {}".format(decoder))
        self.decoder = decoders.__dict__[decoder]()

    def forward(self, image, trimap):
        inp = torch.cat((image, trimap), dim=1)
        embedding, mid_fea = self.encoder(inp)
        alpha, info_dict = self.decoder(embedding, mid_fea)

        return alpha, info_dict


def get_generator(encoder, decoder):
    generator = Generator(encoder=encoder, decoder=decoder)
    return generator


if __name__=="__main__":
    import time
    generator = get_generator(encoder=CONFIG.model.arch.encoder, decoder=CONFIG.model.arch.decoder).cuda().train()
    batch_size = 12
    # generator.eval()
    n_eval = 10
    # pre run the model
    # with torch.no_grad():
    #     for i in range(2):
    #         x = torch.rand(batch_size, 3, 512, 512, device=device)
    #         y = torch.rand(batch_size, 3, 512, 512, device=device)
    #         z = generator(x,y)
    # test without GPU IO

    # x = torch.zeros(batch_size, 3, 512, 512, device=device)
    # y = torch.zeros(batch_size, 1, 512, 512, device=device)
    x = torch.randn(batch_size, 3, 512, 512)
    y = torch.randn(batch_size, 3, 512, 512)
    t = time.time()
    # with torch.no_grad():
    for i in range(n_eval):
        a = generator(x.cuda(),y.cuda())
    torch.cuda.synchronize()
    print(generator.__class__.__name__, 'With IO  \t', f'{(time.time() - t)/n_eval/batch_size:.5f} s')
    print(generator.__class__.__name__, 'FPS \t\t', f'{1/((time.time() - t)/n_eval/batch_size):.5f} s')
    for n, p in generator.named_parameters():
        print(n)
