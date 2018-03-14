from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from scipy import ndimage

img_rows_orig = 600
img_cols_orig = 800

def crop_resize(img, cols, rows, Resize=False):
    """
    Crop the center 400 pixels in the x direction,
    and the lower 400 in the y direction.
    and resize.
    Assume the input image is 3 channel processed data.
    (Size = 600 x 800 x 3)


    :param img: image to be cropped and resized.
    """

    # we crop image from center
    short_edge = 400;
    yy = int(img.shape[0] - short_edge)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:img.shape[1] , xx: xx + short_edge, :]
    img = crop_img

    if Resize:
        img = resize(img, (cols, rows), preserve_range=True)
    return img

def de_crop_resize(imgs, Resize=False):
    short_edge = 400;
    imgs_recover = np.zeros((
        imgs.shape[0], img_rows_orig, img_cols_orig, imgs.shape[3])
    , dtype=np.float32)
    yy = int(img_rows_orig - short_edge)
    xx = int((img_cols_orig - short_edge) / 2)
    
    for i in np.arange(imgs.shape[0]):
        if Resize == True:
            cur_img = resize(imgs[i], (short_edge, short_edge), preserve_range=True)
        else:
            cur_img = imgs[i]
        imgs_recover[i,yy:img_rows_orig,xx:xx+short_edge,:] = cur_img
    
    return imgs_recover

def save_mean_std(mean, std, name):
    mean_std = [mean, std]
    np.save(name, mean_std)

def load_mean_std(name):
    mean_std = np.load(name)
    mean = mean_std[0]
    std = mean_std[1]

    return mean, std

def shift_augmentation(X, X_mask, h_range, w_range):
    X_shift = np.copy(X)
    X_mask_shift = np.copy(X_mask)
    size = X.shape[1:3]
    for i in range(X.shape[0]):
        h_random = np.random.rand() * h_range * 2. - h_range
        w_random = np.random.rand() * w_range * 2. - w_range
        h_shift = int(h_random * size[0])
        w_shift = int(w_random * size[1])
        for j in np.arange(X.shape[3]):
            X_shift[i,:,:,j] = ndimage.shift(X[i,:,:,j], (h_shift, w_shift), order=0)
        if len(X_mask.shape) > 1:
            X_mask_shift[i,:,:] = ndimage.shift(X_mask[i,:,:], (h_shift, w_shift), order=0)
    X_shift = np.concatenate([X_shift, X], axis = 0)
    X_mask_shift = np.concatenate([X_mask_shift, X_mask], axis = 0)
    return X_shift, X_mask_shift