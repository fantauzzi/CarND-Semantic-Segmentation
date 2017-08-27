import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def get_bilinear_filter(filter_shape, upscale_factor):
    # From http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/
    ##filter_shape is [width, height, num_in_channels, num_out_channels]
    kernel_size = filter_shape[1]
    ### Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            ##Interpolation Calculation
            value = (1 - abs((x - centre_location) / upscale_factor)) * (
            1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
                                       shape=weights.shape)
    return bilinear_weights


def upsample_layer(bottom,
                   n_channels, name, upscale_factor):
    # From http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/
    kernel_size = 2 * upscale_factor - upscale_factor % 2
    stride = upscale_factor
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        # Shape of the bottom tensor
        in_shape = tf.shape(bottom)

        h = ((in_shape[1] - 1) * stride) + 1
        w = ((in_shape[2] - 1) * stride) + 1
        new_shape = [in_shape[0], h, w, n_channels]
        output_shape = tf.stack(new_shape)

        filter_shape = [kernel_size, kernel_size, n_channels, n_channels]

        weights = get_bilinear_filter(filter_shape, upscale_factor)
        """deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')"""
        deconv = tf.layers.conv2d_transpose(bottom, n_channels, kernel_size, upscale_factor, 'SAME', kernel_initializer=weights)

    return deconv


def custom_init(shape, dtype=tf.float32, seed=42):
    return tf.random_normal(shape, dtype=dtype, seed=seed)


def conv_1x1(x, num_outputs):
    kernel_size = 1
    stride = 1
    return tf.layers.conv2d(x, num_outputs, kernel_size, stride)


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    helper.maybe_download_pretrained_vgg('data')
    # DONE: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    default_graph=tf.get_default_graph()

    vgg_input_tensor = default_graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = default_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = default_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = default_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = default_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    layer7_1x1 = conv_1x1(vgg_layer7_out, num_classes)
    layer7_upsampled = upsample_layer(layer7_1x1, num_classes, "layer7_upsampled", 2)
    layer4_1x1 = conv_1x1(vgg_layer4_out, num_classes)
    layer4_7_fused = tf.add(layer4_1x1, layer7_upsampled)
    layer4_7_upsampled = upsample_layer(layer4_7_fused, num_classes, "layer4_7_upsampled", 2)
    layer3_1x1 = conv_1x1(vgg_layer3_out, num_classes)
    layer3_4_7_fused = tf.add(layer3_1x1, layer4_7_upsampled)
    layer3_4_7_upsampled= upsample_layer(layer3_4_7_fused, num_classes, "layer3_4_7_upsampled", 8)


    return layer3_4_7_upsampled


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    return None, None, None


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    pass


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
