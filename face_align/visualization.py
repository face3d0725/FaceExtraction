import cv2
import numpy as np


def vis_parsing_maps(im, parsing_anno, stride):
    # Colors for all 20 parts
    part_colors = np.array([[128, 128, 128], [255, 85, 0], [255, 170, 0],
                 [255, 0, 85], [255, 0, 170],
                 [0, 255, 0], [0, 255, 255], [170, 255, 0],
                 [0, 255, 85], [0, 255, 170],
                 [0, 0, 255], [85, 0, 255], [170, 0, 255],
                 [0, 85, 255], [0, 170, 255],
                 [255, 255, 0], [255, 255, 85], [255, 255, 170],
                 [255, 0, 255], [255, 85, 255], [255, 170, 255],
                 [85, 255, 255], [170, 255, 255], [85, 255, 0]],
                dtype=np.uint8)


    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    return vis_im
