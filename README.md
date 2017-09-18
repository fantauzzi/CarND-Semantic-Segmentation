# README
## Semantic Segmentation
In this project, I classify the pixels of a road in images using a Fully Convolutional Network (FCN).

[//]: # (Image References)

[image1]: readme_pics/architecture.png "Network architecture"

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

### Project structure
- `data/` directory for the Kitti Road dataset and the pre-trained VGG16 neural network
- `.gitignore` git configuration file
- `helper.py` source file
- `.idea/` PyCharm project
- `main.py` source file
- `process_tests.py` source file
- `README.md` this file
- `readme_pics/` image for this README
- `requirements.txt` Python dependencies
- `runs/` output of the program, to contain processed images
- `tensorboard/` log files for tensorboard 
  

### Run
Run the following command to run the project:
```
python3 main.py
```
The program train the neural network and then uses it to process the images in the Kitti Road dataset; processed images are saved in a directory under `runs`, with pixels classified as part of the road highlighted in green. 

![processed image][image2]

Log files to visualize the computational graph with tensorboard are saved in the `tensorboard` directory.

Before terminating, the program displays a chart of the loss computed at the end of every epoch, such as in the picture here below.

![chart][image3]


## Neural Network Architecture and Training

The network model is adapted from TODO paper reference!

It starts from a pre-trained VGG16, where the fully connected layers have been removed. The FCN is built on top of the last remaining layer in VGG16 (`Layer 7`, in the picture below), and also with skip-connections with two previous layers of VGG16 (`Layer 3` and `Layer 4`).

![Network architecture][image1]

The picture reports the shape of every layer, assuming a batch size of 2 and an input image of size 576x160 pixels (3 channels).

Weights of the pre-trained VGG16 are used as a starting point to train the overall network, but they are not frozen, and are optimized as part of the training process. 