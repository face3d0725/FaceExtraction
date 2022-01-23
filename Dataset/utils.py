import torch
from torchvision.utils import make_grid
import numpy as np


def get_face_mask(mask):
    mask = torch.from_numpy(mask)
    mask = (mask > 0) * (mask < 11) * (~(mask == 4)) # facial attributes except eyeglasses
    return mask


def tensor2img(tensor):
    shape = tensor.shape
    show = tensor
    if len(shape) == 4:
        if shape[0] == 1:
            show = tensor.squeeze(0)
        else:
            show = make_grid(tensor, nrow=tensor.shape[0], padding=0)

    show = show.detach().permute(1, 2, 0).cpu().numpy()
    if show.max() > 1:
        show = show.astype('uint8')
    elif show.min() < 0:
        show = (show * 128 + 127.5).astype('uint8')
        # show = show*0.5+0.5
        # show = (show * 255).astype('uint8')
    else:
        show = (show * 255).astype('uint8')
    return show
