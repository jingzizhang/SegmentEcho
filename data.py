from __future__ import print_function

import os, glob
import numpy as np

from skimage.io import imsave, imread
from skimage.restoration import denoise_tv_chambolle

data_path = '../LA_area/'

image_rows = 600
image_cols = 800

def create_data():
    train_data_path = os.path.join(data_path)
    image_dirs = os.listdir(train_data_path)

    imgs=[]
    imgs_mask=[]
    imgs_test=[]
    imgs_test_mask=[]
    imgs_test_id=[]

    test_pt = ['pt11','pt12','pt13']

    print('-'*52)
    print('Creating training, validation, and testing images...')
    print('-'*52)
    for image_dir in image_dirs:
        if not '4ch' in image_dir:
            continue
        if 'avi' in image_dir:
            continue
        if 'jpg' in image_dir:
            continue
        cur_data_path = os.path.join(data_path,image_dir)
        images = sorted(os.listdir(cur_data_path))
        
        pt_no = image_dir.split('_')[0] #ptXX_4ch
        if pt_no in test_pt:
            test = True
        else:
            test = False
        
        for image in images:
            if not test:
                if not 'mask0' in image:
                    continue
            
            image_mask_name = image
            pt_no = image.split('_')[0] #ptXX_4ch_XXXmask0.png
            
            if not test:
                img_no = int(image.split('m')[0][-3:])
            else:
                img_no = int(image.split('.')[0][-3:])
                
            if img_no == 1:
                continue
            
            if not test:
                img_id = image.split('m')[0]
            else:
                img_id = image.split('.')[0]
            
            prev_img_name = pt_no + '_4ch_' + format(img_no-1, '03d') + '.png'
            cur_img_name = pt_no + '_4ch_' + format(img_no, '03d') + '.png'
            next_img_name = pt_no + '_4ch_' + format(img_no+1, '03d') + '.png'
            if not os.path.isfile(os.path.join(cur_data_path, next_img_name)):
                continue
            prev_img = imread(os.path.join(cur_data_path, prev_img_name), as_grey=True)
            cur_img = imread(os.path.join(cur_data_path, cur_img_name), as_grey=True)
            next_img = imread(os.path.join(cur_data_path, next_img_name), as_grey=True)

            # 3 ch image. discard the first and the last frames
            # ch0: previous frame
            # ch1: current frame
            # ch2: next frame
            img = np.stack((prev_img,cur_img,next_img), axis=2)
            #img = denoise_tv_chambolle(img, weight=0.1, multichannel=True)
            
            if not test:
                img_mask = imread(os.path.join(cur_data_path, image_mask_name), as_grey=True)

            if test:
                imgs_test.append(img)
                #imgs_test_mask.append(img_mask)
                imgs_test_id.append(img_id)
            else:
                imgs.append(img)
                imgs_mask.append(img_mask)

        print('Done: {}'.format(image_dir))
    print('Loading done.')

    imgs = np.asarray(imgs, dtype=np.float32)
    imgs_mask = np.asarray(imgs_mask, dtype=np.float32)
    imgs_test = np.asarray(imgs_test, dtype=np.float32)
    #imgs_test_mask = np.asarray(imgs_test_mask, dtype=np.float32)

    np.save('imgs_train_312.npy', imgs)
    np.save('imgs_mask_train_312.npy', imgs_mask)
    np.save('imgs_test.npy_312', imgs_test)
    #np.save('imgs_mask_test.npy', imgs_test_mask)
    np.save('imgs_test_id_312.npy', imgs_test_id)

    print('Saving to .npy files done.')
    print('')
    print('imgs_train: {}'.format(imgs.shape))
    print('imgs_mask_train: {}'.format(imgs_mask.shape))
    print('imgs_test: {}'.format(imgs_test.shape))
    #print('imgs_mask_test: {}'.format(imgs_test_mask.shape))
    
if __name__ == '__main__':
    create_data()
    
def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_test_id.npy')
    return imgs_test, imgs_id

def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def load_train_data_312():
    imgs_train = np.load('imgs_train_312.npy')
    imgs_mask_train = np.load('imgs_mask_train_312.npy')
    return imgs_train, imgs_mask_train

def load_train_data_tv():
    imgs_train = np.load('imgs_train_tv.npy')
    imgs_mask_train = np.load('imgs_mask_train_tv.npy')
    return imgs_train, imgs_mask_train

def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

def load_test_data_all():
    imgs_test = np.load('imgs_test_all.npy')
    imgs_id = np.load('imgs_id_test_all.npy')
    return imgs_test, imgs_id

def load_test_data_all_tv():
    imgs_test = np.load('imgs_test_all_tv.npy')
    imgs_id = np.load('imgs_id_test_all.npy')
    return imgs_test, imgs_id

def load_test_data_312():
    imgs_test = np.load('imgs_test_312.npy')
    imgs_id = np.load('imgs_test_id_312.npy')
    return imgs_test, imgs_id

