import os
import cv2
import toml
import argparse
import numpy as np

import torch
from torch.nn import functional as F

import utils
from   utils import CONFIG
import networks


def single_inference(model, image_dict, return_offset=True):

    with torch.no_grad():
        image, trimap = image_dict['image'], image_dict['trimap']
        alpha_shape = image_dict['alpha_shape']
        image = image.cuda()
        trimap = trimap.cuda()
        alpha_pred, info_dict = model(image, trimap)

        if CONFIG.model.trimap_channel == 3:
            trimap_argmax = trimap.argmax(dim=1, keepdim=True)

        alpha_pred[trimap_argmax == 2] = 1
        alpha_pred[trimap_argmax == 0] = 0

        h, w = alpha_shape
        test_pred = alpha_pred[0, 0, ...].data.cpu().numpy() * 255
        test_pred = test_pred.astype(np.uint8)
        test_pred = test_pred[32:h+32, 32:w+32]

        if return_offset:
            short_side = h if h < w else w
            ratio = 512 / short_side
            offset_1 = utils.flow_to_image(info_dict['offset_1'][0][0,...].data.cpu().numpy()).astype(np.uint8)
            # write softmax_scale to offset image
            scale = info_dict['offset_1'][1].cpu()
            offset_1 = cv2.resize(offset_1, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST)
            text = 'unknown: {:.2f}, known: {:.2f}'.format(scale[-1,0].item(), scale[-1,1].item())
            offset_1 = cv2.putText(offset_1, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, thickness=2)

            offset_2 = utils.flow_to_image(info_dict['offset_2'][0][0,...].data.cpu().numpy()).astype(np.uint8)
            # write softmax_scale to offset image
            scale = info_dict['offset_2'][1].cpu()
            offset_2 = cv2.resize(offset_2, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST)
            text = 'unknown: {:.2f}, known: {:.2f}'.format(scale[-1,0].item(), scale[-1,1].item())
            offset_2 = cv2.putText(offset_2, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, thickness=2)

            return test_pred, (offset_1, offset_2)
        else:
            return test_pred, None


def generator_tensor_dict(image_path, trimap_path):
    # read images
    image = cv2.imread(image_path)
    trimap = cv2.imread(trimap_path, 0)
    sample = {'image': image, 'trimap': trimap, 'alpha_shape': trimap.shape}

    # reshape
    h, w = sample["alpha_shape"]
    
    if h % 32 == 0 and w % 32 == 0:
        padded_image = np.pad(sample['image'], ((32,32), (32, 32), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((32,32), (32, 32)), mode="reflect")
        sample['image'] = padded_image
        sample['trimap'] = padded_trimap
    else:
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w
        padded_image = np.pad(sample['image'], ((32,pad_h+32), (32, pad_w+32), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((32,pad_h+32), (32, pad_w+32)), mode="reflect")
        sample['image'] = padded_image
        sample['trimap'] = padded_trimap

    # ImageNet mean & std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    # convert GBR images to RGB
    image, trimap = sample['image'][:,:,::-1], sample['trimap']
    # swap color axis
    image = image.transpose((2, 0, 1)).astype(np.float32)
    trimap[trimap < 85] = 0
    trimap[trimap >= 170] = 2
    trimap[trimap >= 85] = 1
    # normalize image
    image /= 255.

    # to tensor
    sample['image'], sample['trimap'] = torch.from_numpy(image), torch.from_numpy(trimap).to(torch.long)
    sample['image'] = sample['image'].sub_(mean).div_(std)

    if CONFIG.model.trimap_channel == 3:
        sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2, 0, 1).float()
    elif CONFIG.model.trimap_channel == 1:
        sample['trimap'] = sample['trimap'][None, ...].float()
    else:
        raise NotImplementedError("CONFIG.model.trimap_channel can only be 3 or 1")

    # add first channel
    sample['image'], sample['trimap'] = sample['image'][None, ...], sample['trimap'][None, ...]

    return sample

if __name__ == '__main__':
    print('Torch Version: ', torch.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/gca-dist-all-data.toml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/gca-dist-all-data/gca-dist-all-data.pth',
                        help="path of checkpoint")
    parser.add_argument('--image-dir', type=str, default='demo/input_lowres', help="input image dir")
    parser.add_argument('--trimap-dir', type=str, default='demo/trimap_lowres/Trimap3', help="input trimap dir")
    parser.add_argument('--output', type=str, default='demo/pred/Trimap3/', help="output dir")

    # Parse configuration
    args = parser.parse_args()
    with open(args.config) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")

    args.output = os.path.join(args.output, CONFIG.version+'_'+args.checkpoint.split('/')[-1])
    utils.make_dir(args.output)

    # build model
    model = networks.get_generator(encoder=CONFIG.model.arch.encoder, decoder=CONFIG.model.arch.decoder)
    model.cuda()

    # load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

    # inference
    model = model.eval()

    for image_name in os.listdir(args.image_dir):
        # assume image and trimap have the same file name
        image_path = os.path.join(args.image_dir, image_name)
        trimap_path = os.path.join(args.trimap_dir, image_name)
        print('Image: ', image_path, ' Tirmap: ', trimap_path)
        image_dict = generator_tensor_dict(image_path, trimap_path)
        pred, offset = single_inference(model, image_dict)

        cv2.imwrite(os.path.join(args.output, image_name), pred)
        if offset is not None:
            cv2.imwrite(os.path.join(args.output, os.path.splitext(image_name)[0]+'_offset1.png'), offset[0])
            cv2.imwrite(os.path.join(args.output, os.path.splitext(image_name)[0]+'_offset2.png'), offset[1])
