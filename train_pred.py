from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from data import load_train_data, load_test_data, load_test_data_all, load_test_data_all_tv
from data import load_train_data_312, load_test_data_312
from utils import crop_resize, de_crop_resize, save_mean_std, load_mean_std, shift_augmentation

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows_orig = 600
img_cols_orig = 800
img_rows = 400
img_cols = 400
img_ds_rows = 192 # change this to adjust downsampling rate
img_ds_cols = 192 # and this

pred_dir = 'preds_d2' # where the prediction is saved

smooth = 1.

def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_test_id.npy')
    return imgs_test, imgs_id

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_ds_rows, img_ds_cols, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='glorot_uniform')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def preprocess(imgs):
    if len(imgs.shape) == 3:
        imgs = imgs.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[2], 1)

    imgs_p = np.ndarray((imgs.shape[0], img_ds_rows, img_ds_cols, imgs.shape[3]), dtype=np.float32)
    for i in range(imgs.shape[0]):
        imgs_p[i] = crop_resize(imgs[i,:,:,:], img_ds_cols, img_ds_rows, Resize = True)
        #imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    return imgs_p

def train_and_predict_seg():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    print('-'*40)
    print('Loading and preprocessing train data...')
    print('-'*40)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train, imgs_mask_train = shift_augmentation(
        imgs_train, imgs_mask_train, 0.05, 0.1)

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    save_mean_std(mean, std, '3ch_meanstd_d2.npy')

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*40)
    print('Creating and compiling model...')
    print('-'*40)
    seg_model = get_unet()
    seg_model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])
    #seg_model.load_weights('weights_seg_3ch.h5')
    model_checkpoint = ModelCheckpoint('weights_seg_3ch_d2.h5', monitor='val_loss', save_best_only=True)

    print('-'*40)
    print('Fitting model...')
    print('-'*40)
    seg_model.fit(imgs_train, imgs_mask_train, batch_size=55, epochs=20, verbose=1, shuffle=True,
              validation_split=0.1,
              callbacks=[model_checkpoint])
    print('-'*40)
    print('Training Done.')
    print('-'*40)

        ### predict
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)
    #imgs_mask_test_true = np.load('imgs_mask_test.npy')
    #imgs_mask_test_true = preprocess(imgs_mask_test_true)

    mean, std = load_mean_std('3ch_meanstd_d2.npy')

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    seg_model = get_unet()
    seg_model.load_weights('weights_seg_3ch_d2.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = seg_model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test_pred_d2.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    imgs_mask_test_orig = de_crop_resize(imgs_mask_test, Resize = True)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test_orig, imgs_id_test):
        image = ((image[:, :, 0] > 0.5) * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)
    print('-' * 30)
    print('Saving Done.')
    print('-' * 30)

if __name__ == '__main__':
    train_and_predict_seg()
