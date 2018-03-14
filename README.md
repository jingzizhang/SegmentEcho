# SegmentEcho
## Overview
Automatic Heart Chamber Segmentation using Convolutional Neural Networks in 2D Echocardiography 
- Submitted as the Final project for ECE211A, UCLA, Winter 2018 by Prof. Fabien Scalzo

Authors: Jingzi Zhang, Yi Zheng

Department of Electrical and Computer Engineering

University of California, Los Angeles

This project is mainly written in python with Keras.

Due to privacy issues, we cannot disclose our dataset. However, the code should work with common echo data of avi videos with size 600x800.

## Acknowledgement

- We thank Dr. Mirela Tuzovic for providing the dataset and insight that greatly assisted the project.

- The main code framework is borrowed from:

-- https://github.com/jocicmarko/ultrasound-nerve-segmentation

-- https://github.com/jocicmarko/kaggle-dsb2-keras/

- Please refer to LICENSE.md for copyright and license.

## How to use
First, clone the repo to your computer. Go to the folder where this repo is saved.

### Dependencies
Please run

```pip install -r requirements```

to install the dependencies.

### Convert video to png

Save ```video2img.sh``` in the same directory as your echo data.

Run

```./video2img.sh```

### Prepare Data

- Execute python script

1. Change the variable ```data_path ``` in ```data.py ``` to the folder containing your echo data:

2. Execute python script:

```python data.py```

- Or use jupyter notebook

1. Change the variable ```data_path ``` in ```test_data.ipynb ``` to the folder containing your echo data:

2. Execute the notebook ```test_data.ipynb ```

### Preprocess Data, Train and Predict

- Execute python script

1. In ```train_pred.py ```, change the variable ```img_rows ``` and ```img_cols ``` for the cropped size.
Change the variable ```img_ds_rows ``` and ```img_ds_cols ``` for the downsampled size.
Change the variable ```pred_dir ``` to where you want to store the predicted masks.

2. Execute python script:

```python train_pred.py```

- Or use jupyter notebook

1. In ```test_train.ipynb ```, change the variable ```img_rows ``` and ```img_cols ``` for the cropped size.
Change the variable ```img_ds_rows ``` and ```img_ds_cols ``` for the downsampled size.
Change the variable ```pred_dir ``` to where you want to store the predicted masks.

2. Execute the notebook ```test_train.ipynb ```

- The predicted segmentation masks will be saved in pred_dir.





