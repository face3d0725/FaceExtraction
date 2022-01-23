import torch
import torch.nn.functional as F
import math
import cv2


class RandomAffine(object):
    def __init__(self, scale, angle, flip=0.5, translate=0.1):
        self.scale = scale
        self.angle = angle / 180 * math.pi
        self.flip = flip  # probability of left-right flip
        self.translate = translate

    def __call__(self, data):
        # data = {'img': img, 'uv': uv, 'mat_inv': mat_inverse, 'mask': mask}
        img, mask = data['img'], data['mask']
        h, w = img.shape[1:]

        # flip flag
        flip = torch.rand(1).item() < self.flip

        # rotation matrix
        angle = 2 * self.angle * torch.rand(1) - self.angle
        cos = torch.cos(angle).item()
        sin = torch.sin(angle).item()

        s = (1 + 2 * self.scale * torch.rand(1) - self.scale).item()
        t = self.translate * torch.rand(1)
        M = torch.Tensor([
            [s * cos, s * sin, t],
            [-s * sin, s * cos, t]
        ])

        if flip:
            M[0, 0] *= -1
            M[1, 0] *= -1

        grid = F.affine_grid(M.unsqueeze(0), [1, 3, h, w], align_corners=True)
        image = F.grid_sample(img.unsqueeze(0), grid, align_corners=True).squeeze(0)
        mask = F.grid_sample(mask.unsqueeze(0), grid, align_corners=True, mode='nearest').squeeze(0)
        image = torch.clamp(image, 0, 1)
        mask = (mask > 0).type(torch.float32)

        # image = cv2.warpAffine(img, M, (h, w))
        # mask = cv2.warpAffine(mask, M, (h, w), flags=cv2.INTER_NEAREST)

        data = {'img': image, 'mask': mask, }
        return data
