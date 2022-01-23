import cv2
import os
import pathlib
import numpy as np
import pickle
from PIL import Image
from face_align.align import Preprocess
from face_align.visualization import vis_parsing_maps
import tqdm

# Specify the root of related data
root_img = '/media/xn/1TDisk/CelebAMask-HQ/CelebA-HQ-img/'
root_mask = '/media/xn/1TDisk/CelebAMask-HQ/CelebAMask-HQ-mask-anno/'
root_ldmk = '/media/xn/SSD1T/CelebAMask-HQ/ldmk_init/'  # landmark detected by 3ddfa-v2

pth = pathlib.Path(__file__).parent.parent.absolute()
root_img_sv = os.path.join(pth, 'Dataset/CelebA-HQ-align/')
root_mask_sv = os.path.join(pth, 'Dataset/CelebAMask-HQ-align/')
if not os.path.exists(root_img_sv):
    os.mkdir(root_img_sv)
if not os.path.exists(root_mask_sv):
    os.mkdir(root_mask_sv)

atts = ['skin', 'l_brow', 'r_brow', 'eye_g', 'l_eye', 'r_eye', 'nose', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat']


def get_mask(img_name):
    num = int(img_name.split('.')[0])
    folder = str(num // 2000)
    mask = np.zeros((512, 512))
    for l, att in enumerate(atts, 1):
        file_name = ''.join([str(num).rjust(5, '0'), '_', att, '.png'])
        path = os.path.join(root_mask, folder, file_name)
        if os.path.exists(path):
            sep_mask = np.array(Image.open(path).convert('P'))
            mask[sep_mask == 225] = l
    return mask


def get_img(name):
    pth = os.path.join(root_img, name)
    I = cv2.imread(pth)
    I = cv2.resize(I, (512, 512))  # resize the image from 1024x1024 to 512x512
    return I


def get_ldmk(name):
    ldmk_name = name.split('.')[0] + '.pkl'
    pth = os.path.join(root_ldmk, ldmk_name)
    with open(pth, 'rb') as f:
        ldmk = pickle.load(f)
    ldmk = ldmk * 0.5  # original image is 1024x1024 while seg map is 512x512
    return ldmk


def process(img_name):
    img = get_img(img_name)
    seg = get_mask(img_name)
    ldmk = get_ldmk(img_name)
    img, seg, ldmk = Preprocess(img, seg, ldmk)
    return img, seg, ldmk


if __name__ == '__main__':
    for name in tqdm.tqdm(os.listdir(root_img)):
        img, seg, ldmk = process(name)

        # Uncomment for visualization
        img_mask = vis_parsing_maps(img, seg, 1)
        show = np.concatenate((img, img_mask))
        cv2.imshow('seg', show)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

        img_sv_pth = os.path.join(root_img_sv, name)
        cv2.imwrite(img_sv_pth, img)

        seg_name = name.split('.')[0] + '.pkl'
        seg_sv_pth = os.path.join(root_mask_sv, seg_name)
        with open(seg_sv_pth, 'wb') as f:
            pickle.dump(seg, f)

    cv2.destroyAllWindows()
