# Semantic Segmentation
## Introduction
In this project, I classify the pixels of a road in images using a Fully Convolutional Network (FCN).

## Setup
### Required Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
 - [tqdm](https://pypi.python.org/pypi/tqdm)
 - [matplotlib](http://matplotlib.org/)
 
### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder under the project directory.  This will create the folder `data_road` with all the training a test images.

#### Run
Run the following command to run the project:
```
python3 main.py
```

## Neural Network Architecture

The network model is adapted from TODO paper reference!

It starts from a pre-trained VGG16, where them fully connected layers have been removed. The FCN is built on top of the last remaining layer in VGG16, and also with skip-connections with two previous layers of VGG16.

The drawing below illustrates how the FCN has been retrofitted on a pre-trained VGG16.