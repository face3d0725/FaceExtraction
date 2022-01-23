from torchvision import transforms as TF
import torch.utils.data as Data
from Dataset.utils import get_face_mask
from Dataset.transforms import RandomAffine
import torch
import os
import glob
import cv2
import random
import pathlib
from PIL import Image
import pickle

pth = pathlib.Path(__file__).parent.parent.absolute()
img_root = os.path.join(pth, 'Dataset', 'CelebA-HQ-align')
mask_root = os.path.join(pth, 'Dataset', 'CelebAMask-HQ-align')
occ_root = os.path.join(pth, 'Dataset', 'FaceOcc')

cofw_img_root = os.path.join(pth, 'Dataset', 'FaceOcc', 'COFW_test/img')
cofw_mask_root = os.path.join(pth, 'Dataset', 'FaceOcc', 'COFW_test/mask')

os.chdir(occ_root)
FaceOcc = {'celeba': glob.glob('CelebAHQ/*'),
           'ffhq': glob.glob('ffhq/*'),
           'internet': glob.glob('internet/*/*'),
           'texture': glob.glob('texture/*')}

os.chdir(pth)


class OcclusionGenerator():
    def __init__(self):
        self.transform = TF.Compose([
            TF.RandomAffine(degrees=10, translate=(0.1, 0.1),
                            scale=(0.8, 1.2),
                            shear=(-10, 10, -0.1, 0.1)),
            TF.RandomHorizontalFlip(0.5),
            TF.ToTensor()
        ])

        self.transform_mask = TF.Compose([TF.RandomAffine(degrees=5, translate=(0.0, 0.1), scale=(0.8, 1),
                                                          shear=(-5, 5, -0.1, 0.1)),
                                          TF.RandomHorizontalFlip(0.5),
                                          TF.ToTensor()])

        self.transform_color = TF.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
        self.occ_lst = FaceOcc['internet'] * 5 + FaceOcc['celeba'] + FaceOcc['ffhq']
        self.texture_lst = FaceOcc['texture']
        self.root = occ_root

    def __call__(self):
        img_name = random.choice(self.occ_lst)
        pth = os.path.join(self.root, img_name)
        o = Image.open(pth)
        if 'mask' in img_name:
            o = self.transform_mask(o)
        else:
            o = self.transform(o)
        rgb, mask = o[:3], o[3:]
        rand_tex = random.random()
        if 'mask' in img_name:
            rand_tex = rand_tex > 0.1
        else:
            rand_tex = rand_tex > 0.9
        if rand_tex:
            tex_name = random.choice(self.texture_lst)
            tex_pth = os.path.join(self.root, tex_name)
            tex = Image.open(tex_pth)
            tex = self.transform(tex)
            rgb = tex
        rgb = self.transform_color(rgb)
        return rgb, mask


class FaceMask(Data.Dataset):
    def __init__(self):
        self.img_root = img_root
        self.mask_root = mask_root
        self.img_list = self.get_image_list()
        self.img_list_real = FaceOcc['celeba'] + FaceOcc['ffhq']  # directly use occ data for training
        self.occ_gen = OcclusionGenerator()
        self.to_tensor = TF.Compose([TF.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                     TF.ToTensor()])
        self.transform = RandomAffine(0.2, 20, 0.5)

    def get_image_list(self):
        image_list = os.listdir(self.img_root)  # images in CelebA-HQ
        occ_list = os.listdir(os.path.join(occ_root, 'CelebAHQ'))  # occluded images in CelebA-HQ
        occ_list = [n.split('.')[0] + '.jpg' for n in occ_list]  # .png to .jpg
        image_list = list(set(image_list) - set(occ_list))
        return image_list

    def __len__(self):
        return len(self.img_list)

    def get_aug_data(self, name):
        I = Image.open(os.path.join(self.img_root, name))
        I = self.to_tensor(I)
        pkl_name = name.split('.')[0] + '.pkl'
        with open(os.path.join(self.mask_root, pkl_name), 'rb') as f:
            mask = pickle.load(f)
        mask = get_face_mask(mask).unsqueeze(0)
        occ, mask_occ = self.occ_gen()
        I = occ * mask_occ + I * (1 - mask_occ)
        mask = mask * (1 - mask_occ)
        data = {'img': I, 'mask': mask}
        data = self.transform(data)
        I, mask = data['img'], data['mask']
        return I, mask

    def get_real_data(self):
        name = random.choice(self.img_list_real)
        I = Image.open(os.path.join(occ_root, name))
        I = self.to_tensor(I)
        rgb, mask_occ = I.split([3, 1], dim=0)
        pkl_name = name.split('/')[1].split('.')[0] + '.pkl'
        if 'ffhq' in name:
            mask_pth = os.path.join(occ_root, 'ffhqMask', pkl_name)
        else:
            mask_pth = os.path.join(self.mask_root, pkl_name)
        with open(mask_pth, 'rb') as f:
            mask = pickle.load(f)

        mask = get_face_mask(mask) * (1 - mask_occ)
        data = {'img': rgb, 'mask': mask}
        data = self.transform(data)
        I, mask = data['img'], data['mask']
        return I, mask

    def __getitem__(self, idx):
        p = random.random()
        if p < .3:
            I, mask = self.get_real_data()
        else:
            name = self.img_list[idx]
            I, mask = self.get_aug_data(name)
        return I, mask


class COFW_test(Data.Dataset):
    def __init__(self):
        self.img_root = cofw_img_root
        self.mask_root = cofw_mask_root
        self.to_tensor = TF.ToTensor()
        self.img_lst = os.listdir(self.img_root)

    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, idx):
        name = self.img_lst[idx]
        I = Image.open(os.path.join(self.img_root, name))
        I = self.to_tensor(I)
        mask_name = name.split('.')[0] + '.png'
        mask = cv2.imread(os.path.join(self.mask_root, mask_name), 0)
        mask = self.to_tensor(mask)
        return I, mask


class InputFetcher(object):
    def __init__(self, batch_size=8, name='celeba', device='cuda:0'):
        self.device = device
        if name == 'train':
            self.dataset = FaceMask()
        elif name == 'test':
            self.dataset = COFW_test()
        # elif name == 'test':
        #     self.dataset = CelebA_test()
        else:
            raise NotImplementedError('dataset not implemented')
        self.batch_size = batch_size
        self.iter = self.get_iter()

    def get_iter(self):
        loader = Data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                 drop_last=True)
        return iter(loader)

    def __len__(self):
        return len(self.dataset)

    def __next__(self):
        try:
            out = next(self.iter)

        except StopIteration:
            self.iter = self.get_iter()
            out = next(self.iter)

        out = [o.to(self.device) for o in out]
        return out[0], out[1]


if __name__ == '__main__':
    import numpy as np
    from Dataset.utils import tensor2img


    def show_mask(I, mask):
        mask = tensor2img(mask) // 255
        I = I * (1 - mask) + (I * mask * 0.5) + (mask * np.array([0, 85, 255]) * 0.5)
        I = I.astype('uint8')
        return I


    fetcher = InputFetcher(batch_size=4, name='test')
    for idx in range(1000):
        I, mask = next(fetcher)
        I = torch.clamp(I, 0, 1)
        face = I * mask
        I = tensor2img(I)
        show_face = show_mask(I, mask)
        face = tensor2img(face)
        show = np.concatenate((I, show_face, face), axis=0)
        cv2.imshow('show', show[..., ::-1])
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
