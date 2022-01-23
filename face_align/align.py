from scipy.io import loadmat
import numpy as np
import cv2
import pathlib
from PIL import Image
import os


def load_lm3d():
    pth = os.path.join(pathlib.Path(__file__).parent.parent, 'face_align/similarity_Lm3D_all.mat')
    Lm3D = loadmat(pth)
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(Lm3D[lm_idx[[3, 4]], :], 0),
                     Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D


def POS(xp, x):
    npts = xp.shape[0]
    if npts == 68:
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        xp = np.stack([xp[lm_idx[0], :], np.mean(xp[lm_idx[[1, 2]], :], 0), np.mean(xp[lm_idx[[3, 4]], :], 0),
                       xp[lm_idx[5], :], xp[lm_idx[6], :]], axis=0)
        xp = xp[[1, 2, 0, 3, 4], :]
        npts = 5
    A = np.zeros([2 * npts, 8])
    x = np.concatenate((x, np.ones((npts, 1))), axis=1)
    A[0:2 * npts - 1:2, 0:4] = x

    A[1:2 * npts:2, 4:] = x

    b = np.reshape(xp, [-1, 1])

    k, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.stack([sTx, sTy], axis=0)

    return t, s


def process_img(img, seg, t, s, target_size=256):
    h0, w0 = img.shape[:2]
    scale = 116. / s
    dx = -(t[0, 0] * scale - target_size / 2)
    dy = -((h0 - t[1, 0]) * scale - target_size / 2)
    mat = np.array([[scale, 0, dx],
                    [0, scale, dy]])

    # process segmentation mask,
    # the cv2.warpAffine have bug for nearest sampling, the edges are inaccurate,
    # so we use the implementation of Image
    mat_extend = np.array([[scale, 0, dx],
                           [0, scale, dy],
                           [0, 0, 1]])
    coeff = np.linalg.inv(mat_extend).flatten()[:6]
    seg = Image.fromarray(seg)
    seg_affine = seg.transform((target_size, target_size), Image.AFFINE, coeff, resample=Image.NEAREST)
    seg_affine = np.array(seg_affine)

    corners = np.array([[0, 0, 1], [w0 - 1, h0 - 1, 1]])
    new_corners = (corners @ mat.T).astype('int32')
    pad_left = max(new_corners[0, 0], 0)
    pad_top = max(new_corners[0, 1], 0)
    pad_right = min(new_corners[1, 0], target_size - 1)
    pad_bottom = min(new_corners[1, 1], target_size - 1)
    mask = np.zeros((target_size, target_size, 3))
    mask[:pad_top, :, :] = 1
    mask[pad_bottom:, :, :] = 1
    mask[:, :pad_left, :] = 1
    mask[:, pad_right:, :] = 1
    img_affine = cv2.warpAffine(img, mat, (target_size, target_size), borderMode=cv2.BORDER_REFLECT_101)
    img_affine = img_affine.astype('float32') * (1 - mask) + cv2.blur(img_affine, (10, 10)) * mask
    img_affine = img_affine.astype('uint8')

    return img_affine, seg_affine, mat


def show_ldmk(img, lm):
    for pt in lm:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), 1)
    return img


def Preprocess(img, seg, lm):
    h0, w0 = img.shape[:2]
    lm_ = np.stack([lm[:, 0], h0 - 1 - lm[:, 1]], axis=1)
    t, s = POS(lm_, lm3D)
    img_new, seg_new, mat = process_img(img, seg, t, s)
    lm_affine = np.concatenate((lm[:, :2], np.ones((lm.shape[0], 1))), axis=1)
    lm_affine = lm_affine @ mat.T
    return img_new, seg_new, lm_affine


lm3D = load_lm3d()
