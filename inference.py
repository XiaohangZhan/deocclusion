import numpy as np
import cv2

import torch
import torch.nn as nn

import utils
import pdb
from skimage.morphology import convex_hull

def to_eraser(inst, bbox, newbbox):
    inst = inst.squeeze(0).numpy()
    final_h, final_w = inst.shape[:2]
    w, h = bbox.numpy()[2:]
    inst = cv2.resize(inst, (w, h), interpolation=cv2.INTER_LINEAR)
    offbbox = [newbbox[0] - bbox[0], newbbox[1] - bbox[1], newbbox[2], newbbox[3]]
    eraser = utils.crop_padding(inst, offbbox, pad_value=(0,))
    eraser = cv2.resize(eraser, (final_w, final_h), interpolation=cv2.INTER_NEAREST)
    #eraser = (eraser >= 0.5).astype(inst.dtype)
    return torch.from_numpy(eraser).unsqueeze(0)

def get_eraser(inst_ind, idx, bbox, input_size):
    inst_ind = inst_ind.numpy()
    bbox = bbox.numpy().tolist()
    eraser = cv2.resize(utils.crop_padding(inst_ind, bbox, pad_value=(0,)),
        (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    eraser = (eraser == idx + 1)
    return torch.from_numpy(eraser.astype(np.float32)).unsqueeze(0)

def net_forward(model, image, inmodal_patch, eraser, use_rgb, th):
    if use_rgb:
        image = torch.from_numpy(image.transpose((2,0,1)).astype(np.float32)).unsqueeze(0)
        image = image.cuda()
    inmodal_patch = torch.from_numpy(inmodal_patch.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    with torch.no_grad():
        if eraser is not None:
            eraser = torch.from_numpy(eraser.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
            if use_rgb:
                output = model.model(torch.cat([inmodal_patch, eraser], dim=1), image)
            else:
                output = model.model(torch.cat([inmodal_patch, eraser], dim=1))
        else:
            if use_rgb:
                output = model.model(torch.cat([inmodal_patch], dim=1), image)
            else:
                output = model.model(inmodal_patch)
        output = nn.functional.softmax(output, dim=1)
    output.detach_()
    return (output[0,1,:,:] > th).cpu().numpy().astype(np.uint8)

def net_forward_ordernet(model, image, inmodal1, inmodal2, use_rgb):
    if use_rgb:
        image = torch.from_numpy(image.transpose((2,0,1)).astype(np.float32)).unsqueeze(0).cuda()
    inmodal1 = torch.from_numpy(
        inmodal1.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    inmodal2 = torch.from_numpy(
        inmodal2.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    with torch.no_grad():
        if use_rgb:
            output1 = nn.functional.softmax(model.model(
                torch.cat([inmodal1, inmodal2, image], dim=1)))
            output2 = nn.functional.softmax(model.model(
                torch.cat([inmodal2, inmodal1, image], dim=1)))
        else:
            output1 = nn.functional.softmax(model.model(
                torch.cat([inmodal1, inmodal2], dim=1)))
            output2 = nn.functional.softmax(model.model(
                torch.cat([inmodal2, inmodal1], dim=1)))
        output1.detach_()
        output2.detach_()
        prob = (output1[:,1] + output2[:,0]) / 2 # average results
        return prob.cpu().numpy().item() > 0.5 # whether 1 over 2
        
def recover_mask(mask, bbox, h, w, interp):
    size = bbox[2]
    if interp == 'linear':
        mask = (cv2.resize(mask.astype(np.float32), (size, size),
            interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    else:
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    woff, hoff = bbox[0], bbox[1]
    newbbox = [-woff, -hoff, w, h]
    return utils.crop_padding(mask, newbbox, pad_value=(0,))

def resize_mask(mask, size, interp):
    if interp == 'linear':
        return (cv2.resize(
            mask.astype(np.float32), (size, size),
            interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    else:
        return cv2.resize(
            mask, (size, size), interpolation=cv2.INTER_NEAREST)

def infer_amodal_hull(inmodal, bboxes, order_matrix, order_grounded=True):
    amodal = []
    num = inmodal.shape[0]
    for i in range(num):
        m = inmodal[i]
        hull = convex_hull.convex_hull_image(m).astype(np.uint8)
        if order_grounded:
            assert order_matrix is not None
            ancestors = get_ancestors(order_matrix, i)
            eraser = (inmodal[ancestors, ...].sum(axis=0) > 0).astype(np.uint8) # union
            hull[(eraser == 0) & (m == 0)] = 0
        amodal.append(hull)
    return amodal

def infer_order_hull(inmodal):
    num = inmodal.shape[0]
    order_matrix = np.zeros((num, num), dtype=np.int)
    occ_value_matrix = np.zeros((num, num), dtype=np.float32)
    for i in range(num):
        for j in range(i + 1, num):
            if bordering(inmodal[i], inmodal[j]):
                amodal_i = convex_hull.convex_hull_image(inmodal[i])
                amodal_j = convex_hull.convex_hull_image(inmodal[j])
                occ_value_matrix[i, j] = ((amodal_i > inmodal[i]) & (inmodal[j] == 1)).sum()
                occ_value_matrix[j, i] = ((amodal_j > inmodal[j]) & (inmodal[i] == 1)).sum()
    order_matrix[occ_value_matrix > occ_value_matrix.transpose()] = -1
    order_matrix[occ_value_matrix < occ_value_matrix.transpose()] = 1
    order_matrix[(occ_value_matrix == 0) & (occ_value_matrix == 0).transpose()] = 0
    return order_matrix

def infer_order_area(inmodal, above='larger'):
    num = inmodal.shape[0]
    order_matrix = np.zeros((num, num), dtype=np.int)
    for i in range(num):
        for j in range(i + 1, num):
            if bordering(inmodal[i], inmodal[j]):
                area_i = inmodal[i].sum()
                area_j = inmodal[j].sum()
                if (area_i < area_j and above == 'larger') or \
                   (area_i >= area_j and above == 'smaller'):
                    order_matrix[i, j] = -1 # i occluded by j
                    order_matrix[j, i] = 1
                else:
                    order_matrix[i, j] = 1
                    order_matrix[j, i] = -1
    return order_matrix

def infer_order_yaxis(inmodal):
    num = inmodal.shape[0]
    order_matrix = np.zeros((num, num), dtype=np.int)
    for i in range(num):
        for j in range(i + 1, num):
            if bordering(inmodal[i], inmodal[j]):
                center_i = [coord.mean() for coord in np.where(inmodal[i] == 1)] # y, x
                center_j = [coord.mean() for coord in np.where(inmodal[j] == 1)] # y, x
                if center_i[0] < center_j[0]: # i higher than j in y axis
                    order_matrix[i, j] = -1 # i occluded by j
                    order_matrix[j, i] = 1
                else:
                    order_matrix[i, j] = 1
                    order_matrix[j, i] = -1
    return order_matrix

def infer_order_sup(model, image, inmodal, bboxes, input_size=256, use_rgb=True):
    num = inmodal.shape[0]
    order_matrix = np.zeros((num, num), dtype=np.int)
    for i in range(num):
        for j in range(i + 1, num):
            if bordering(inmodal[i], inmodal[j]):
                bbox = utils.combine_bbox(bboxes[(i,j), :])
                centerx = bbox[0] + bbox[2] / 2.
                centery = bbox[1] + bbox[3] / 2.
                size = max([np.sqrt(bbox[2] * bbox[3] * 2.), bbox[2] * 1.1, bbox[3] * 1.1])
                new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), \
                            int(size), int(size)]
                image_patch = cv2.resize(utils.crop_padding(
                    image, new_bbox, pad_value=(0,0,0)),
                    (input_size, input_size), interpolation=cv2.INTER_CUBIC)
                modal_i_patch = resize_mask(utils.crop_padding(
                    inmodal[i], new_bbox, pad_value=(0,)),
                    input_size, 'nearest')
                modal_j_patch = resize_mask(utils.crop_padding(
                    inmodal[j], new_bbox, pad_value=(0,)),
                    input_size, 'nearest')
                if np.random.rand() > 0.5: # randomize the input order
                    j_over_i = net_forward_ordernet(
                        model, image_patch, modal_j_patch, modal_i_patch, use_rgb)
                else:
                    j_over_i = not net_forward_ordernet(
                        model, image_patch, modal_i_patch, modal_j_patch, use_rgb)
                if j_over_i:
                    order_matrix[i, j] = -1
                    order_matrix[j, i] = 1
                else:
                    order_matrix[i, j] = 1
                    order_matrix[j, i] = -1

    return order_matrix

def infer_order(model, image, inmodal, category, bboxes, use_rgb=True, th=0.5, dilate_kernel=0, input_size=None, min_input_size=32, interp='nearest', debug_info=False):
    '''
    image: HW3, inmodal: NHW, category: N, bboxes: N4
    '''
    deal_with_fullcover = False
    num = inmodal.shape[0]
    order_matrix = np.zeros((num, num), dtype=np.int)
    ind = []
    if deal_with_fullcover:
        fullcover_inds = []
    for i in range(num):
        for j in range(i + 1, num):
            if bordering(inmodal[i], inmodal[j]):
                ind.append([i, j])
                ind.append([j, i])
            if deal_with_fullcover:
                fullcover = fullcovering(inmodal[i], inmodal[j], bboxes[i], bboxes[j])
                if fullcover == 1:
                    fullcover_inds.append([i, j])
                elif fullcover == 2:
                    fullcover_inds.append([j, i])
    pairnum = len(ind)
    if pairnum == 0:
        return order_matrix
    ind = np.array(ind)
    eraser_patches = []
    inmodal_patches = []
    amodal_patches = []
    ratios = []
    for i in range(pairnum):
        tid = ind[i, 0]
        eid = ind[i, 1]
        image_patch = utils.crop_padding(image, bboxes[tid], pad_value=(0,0,0))
        inmodal_patch = utils.crop_padding(inmodal[tid], bboxes[tid], pad_value=(0,))
        if input_size is not None:
            newsize = input_size
        elif min_input_size > bboxes[tid,2]:
            newsize = min_input_size
        else:
            newsize = None
        if newsize is not None:
            inmodal_patch = resize_mask(inmodal_patch, newsize, interp)
        eraser = utils.crop_padding(inmodal[eid], bboxes[tid], pad_value=(0,))
        if newsize is not None:
            eraser = resize_mask(eraser, newsize, interp)
        if dilate_kernel > 0:
            eraser = cv2.dilate(eraser, np.ones((dilate_kernel, dilate_kernel), np.uint8),
                                iterations=1)
        # erase inmodal
        inmodal_patch[eraser == 1] = 0
        # gather
        inmodal_patches.append(inmodal_patch)
        eraser_patches.append(eraser)
        amodal_patches.append(net_forward(
            model, image_patch, inmodal_patch * category[tid], eraser, use_rgb, th))
        ratios.append(1. if newsize is None else bboxes[tid,2] / float(newsize))

    occ_value_matrix = np.zeros((num, num), dtype=np.float32)
    for i, idx in enumerate(ind):
        occ_value_matrix[idx[0], idx[1]] = (
            ((amodal_patches[i] > inmodal_patches[i]) & (eraser_patches[i] == 1)
             ).sum() * (ratios[i] ** 2))
    order_matrix[occ_value_matrix > occ_value_matrix.transpose()] = -1
    order_matrix[occ_value_matrix < occ_value_matrix.transpose()] = 1
    order_matrix[(occ_value_matrix == 0) & (occ_value_matrix == 0).transpose()] = 0
    if deal_with_fullcover:
        for fc in fullcover_inds:
            assert order_matrix[fc[0], fc[1]] == 0
            order_matrix[fc[0], fc[1]] = -1
            order_matrix[fc[1], fc[0]] = 1
    if debug_info:
        return order_matrix, ind, inmodal_patches, eraser_patches, amodal_patches
    else:
        return order_matrix

def bordering(a, b):
    dilate_kernel = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=np.uint8)
    a_dilate = cv2.dilate(a.astype(np.uint8), dilate_kernel, iterations=1)
    return np.any((a_dilate == 1) & b)

def bbox_in(box1, box2):
    l1, u1, r1, b1 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    l2, u2, r2, b2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
    if l1 >= l2 and u1 >= u2 and r1 <= r2 and b1 <= b2:
        return True
    else:
        return False

def fullcovering(mask1, mask2, box1, box2):
    if not (mask1 == 0).all() and not (mask2 == 0).all():
        return 0
    if (mask1 == 0).all() and bbox_in(box1, box2): # 1 covered by 2
        return 1
    elif (mask2 == 0).all() and bbox_in(box2, box1):
        return 2
    else:
        return 0

def infer_gt_order(inmodal, amodal):
    #inmodal = inmodal.numpy()
    #amodal = amodal.numpy()
    num = inmodal.shape[0]
    gt_order_matrix = np.zeros((num, num), dtype=np.int)
    for i in range(num):
        for j in range(i + 1, num):
            if not bordering(inmodal[i], inmodal[j]):
                continue
            occ_ij = ((inmodal[i] == 1) & (amodal[j] == 1)).sum()
            occ_ji = ((inmodal[j] == 1) & (amodal[i] == 1)).sum()
            #assert not (occ_ij > 0 and occ_ji > 0) # assertion error, why?
            if occ_ij == 0 and occ_ji == 0: # bordering but not occluded
                continue
            gt_order_matrix[i, j] = 1 if occ_ij >= occ_ji else -1
            gt_order_matrix[j, i] = -gt_order_matrix[i, j]
    return gt_order_matrix
 
           
def eval_order(order_matrix, gt_order_matrix):
    inst_num = order_matrix.shape[0]
    allpair_true = ((order_matrix == gt_order_matrix).sum() - inst_num) / 2
    allpair = (inst_num  * inst_num - inst_num) / 2

    occpair_true = ((order_matrix == gt_order_matrix) & (gt_order_matrix != 0)).sum() / 2
    occpair = (gt_order_matrix != 0).sum() / 2

    err = np.where(order_matrix != gt_order_matrix)
    gt_err = gt_order_matrix[err]
    pred_err = order_matrix[err]
    show_err = np.concatenate([np.array(err).T + 1, gt_err[:,np.newaxis], pred_err[:,np.newaxis]], axis=1)
    return allpair_true, allpair, occpair_true, occpair, show_err

def get_neighbors(graph, idx):
    return np.where(graph[idx,:] != 0)[0]

def get_ancestors(graph, idx):
    is_ancestor = np.zeros((graph.shape[0],), dtype=np.bool)
    visited = np.zeros((graph.shape[0],), dtype=np.bool)
    queue = {idx}
    while len(queue) > 0:
        q = queue.pop()
        if visited[q]:
            continue # incase there exists cycles.
        visited[q] = True
        new_ancestor = np.where(graph[q, :] == -1)[0]
        is_ancestor[new_ancestor] = True
        queue.update(set(new_ancestor.tolist()))
    is_ancestor[idx] = False
    return np.where(is_ancestor)[0]

def infer_instseg(model, image, category, bboxes, new_bboxes, input_size, th, rgb=None):
    num = bboxes.shape[0]
    seg_patches = []
    for i in range(num):
        rel_bbox = [bboxes[i,0] - new_bboxes[i,0],
                    bboxes[i,1] - new_bboxes[i,1], bboxes[i,2], bboxes[i,3]]
        bbox_mask = np.zeros((new_bboxes[i,3], new_bboxes[i,2]), dtype=np.uint8)
        bbox_mask[rel_bbox[1]:rel_bbox[1]+rel_bbox[3], rel_bbox[0]:rel_bbox[0]+rel_bbox[2]] = 1
        bbox_mask = cv2.resize(bbox_mask, (input_size, input_size),
            interpolation=cv2.INTER_NEAREST)
        bbox_mask_tensor = torch.from_numpy(
            bbox_mask.astype(np.float32) * category[i]).unsqueeze(0).unsqueeze(0).cuda()
        image_patch = cv2.resize(utils.crop_padding(image, new_bboxes[i], pad_value=(0,0,0)),
            (input_size, input_size), interpolation=cv2.INTER_CUBIC)
        image_tensor = torch.from_numpy(
            image_patch.transpose((2,0,1)).astype(np.float32)).unsqueeze(0).cuda() # 13HW
        with torch.no_grad():
            output = model.model(torch.cat([image_tensor, bbox_mask_tensor], dim=1)).detach()
        if output.shape[2] != image_tensor.shape[2]:
            output = nn.functional.interpolate(
                output, size=image_tensor.shape[2:4],
                mode="bilinear", align_corners=True) # 12HW
        output = nn.functional.softmax(output, dim=1) # 12HW
        if rgb is not None:
            prob = output[0,...].cpu().numpy() # 2HW
            rgb_patch = cv2.resize(utils.crop_padding(rgb, new_bboxes[i], pad_value=(0,0,0)),
                (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            prob_crf = np.array(utils.densecrf(prob, rgb_patch)).reshape(*prob.shape)
            pred = (prob_crf[1,:,:] > th).astype(np.uint8) # HW
        else:
            pred = (output[0,1,:,:] > th).cpu().numpy().astype(np.uint8) # HW
        seg_patches.append(pred)
    return seg_patches

def infer_amodal_sup(model, image, inmodal, category, bboxes, use_rgb=True, th=0.5,
                     input_size=None, min_input_size=16, interp='nearest', debug_info=False):
    num = inmodal.shape[0]
    inmodal_patches = []
    amodal_patches = []
    for i in range(num):
        image_patch = utils.crop_padding(image, bboxes[i], pad_value=(0,0,0))
        inmodal_patch = utils.crop_padding(inmodal[i], bboxes[i], pad_value=(0,))
        if input_size is not None:
            newsize = input_size
        elif min_input_size > bboxes[i,2]:
            newsize = min_input_size
        else:
            newsize = None
        if newsize is not None:
            inmodal_patch = resize_mask(inmodal_patch, newsize, interp)
        inmodal_patches.append(inmodal_patch)
        amodal_patches.append(net_forward(
            model, image_patch, inmodal_patch * category[i], None, use_rgb, th))
    if debug_info:
        return inmodal_patches, amodal_patches
    else:
        return amodal_patches
        
def infer_amodal(model, image, inmodal, category, bboxes, order_matrix,
                use_rgb=True, th=0.5, dilate_kernel=0,
                input_size=None, min_input_size=16, interp='nearest',
                order_grounded=True, debug_info=False):
    num = inmodal.shape[0]
    inmodal_patches = []
    eraser_patches = []
    amodal_patches = []
    for i in range(num):
        if order_grounded:
            ancestors = get_ancestors(order_matrix, i)
        else:
            ancestors = get_neighbors(order_matrix, i)
        image_patch = utils.crop_padding(image, bboxes[i], pad_value=(0,0,0))
        inmodal_patch = utils.crop_padding(inmodal[i], bboxes[i], pad_value=(0,))
        if input_size is not None: # always
            newsize = input_size
        elif min_input_size > bboxes[i,2]:
            newsize = min_input_size
        else:
            newsize = None
        if newsize is not None:
            inmodal_patch = resize_mask(inmodal_patch, newsize, interp)

        eraser = (inmodal[ancestors,...].sum(axis=0) > 0).astype(np.uint8) # union
        eraser = utils.crop_padding(eraser, bboxes[i], pad_value=(0,))
        if newsize is not None:
            eraser = resize_mask(eraser, newsize, interp)
        if dilate_kernel > 0:
            eraser = cv2.dilate(eraser, np.ones((dilate_kernel, dilate_kernel), np.uint8),
                                iterations=1)
        # erase inmodal
        inmodal_patch[eraser == 1] = 0
        # gather
        inmodal_patches.append(inmodal_patch)
        eraser_patches.append(eraser)
        amodal_patches.append(net_forward(
            model, image_patch, inmodal_patch * category[i], eraser, use_rgb, th))
    if debug_info:
        return inmodal_patches, eraser_patches, amodal_patches
    else:
        return amodal_patches


def patch_to_fullimage(patches, bboxes, height, width, interp):
    amodals = []
    for patch, bbox in zip(patches, bboxes):
        amodals.append(recover_mask(patch, bbox, height, width, interp))
    return np.array(amodals)
