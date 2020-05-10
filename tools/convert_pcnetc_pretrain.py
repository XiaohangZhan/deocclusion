import torch

params = torch.load('pretrains/partialconv.pth')['model']

involved_keys = ['enc_1.conv.input_conv.weight',
                 'enc_1.conv.mask_conv.weight',
                 'dec_1.conv.input_conv.weight',
                 'dec_1.conv.mask_conv.weight']

for ik in involved_keys:
    w = params[ik]
    zeros = torch.zeros((w.size(0), 1, w.size(2), w.size(3)), dtype=torch.float32)
    neww = torch.cat([w, zeros], dim=1)
    params[ik] = neww

torch.save(params, 'pretrains/partialconv_input_ch4.pth')
