import os.path
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer
import helper
import warnings
from distutils.version import LooseVersion
# import project_tests as tests
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import scipy
# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def upsample_layer(bottom,
                   n_channels, name, upscale_factor):
    # Adapted from http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/
    kernel_size = 2 * upscale_factor - upscale_factor % 2
    stride = upscale_factor
    # strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        # Shape of the bottom tensor
        # in_shape = tf.shape(bottom)

        # h = ((in_shape[1] - 1) * stride) + 1
        # w = ((in_shape[2] - 1) * stride) + 1
        # new_shape = [in_shape[0], h, w, n_channels]
        # output_shape = tf.stack(new_shape)

        # filter_shape = [kernel_size, kernel_size, n_channels, n_channels]

        weights = tf.truncated_normal_initializer(stddev=0.01)
        deconv = tf.layers.conv2d_transpose(inputs=bottom,
                                            filters=n_channels,
                                            kernel_size=kernel_size,
                                            strides=upscale_factor,
                                            padding='same',
                                            kernel_initializer=weights,
                                            kernel_regularizer=l2_regularizer(1e-3))

    return deconv


def conv_1x1(x, num_outputs):
    return tf.layers.conv2d(inputs=x,
                            filters=num_outputs,
                            kernel_size=1,
                            strides=1,
                            padding='same',
                            kernel_regularizer=l2_regularizer(1e-3))


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
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    default_graph = tf.get_default_graph()

    vgg_input_tensor = default_graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = default_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = default_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = default_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = default_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor


# tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # DONE: Implement function

    """vgg_layer7_out = tf.Print(vgg_layer7_out, [tf.shape(vgg_layer7_out)], message= "Shape of vgg_layer7_out:", summarize=10, first_n=1)
    vgg_layer4_out = tf.Print(vgg_layer4_out, [tf.shape(vgg_layer4_out)], message="Shape of vgg_layer4_out_out:",
                              summarize=10, first_n=1)
    vgg_layer3_out = tf.Print(vgg_layer3_out, [tf.shape(vgg_layer3_out)], message="Shape of vgg_layer3_out:",
                              summarize=10, first_n=1)"""

    layer7_1x1 = conv_1x1(vgg_layer7_out, num_classes)
    # layer7_1x1 = tf.Print(layer7_1x1, [tf.shape(layer7_1x1)], message= "Shape of layer7_1x1:", summarize=10, first_n=1)

    layer7_upsampled = upsample_layer(layer7_1x1, num_classes, "layer7_upsampled", 2)
    # layer7_upsampled = tf.Print(layer7_upsampled, [tf.shape(layer7_upsampled)], message= "Shape of layer7_upsampled:", summarize=10, first_n=1)

    layer4_1x1 = conv_1x1(vgg_layer4_out, num_classes)
    # layer4_1x1 = tf.Print(layer4_1x1, [tf.shape(layer4_1x1)], message= "Shape of layer4_1x1:", summarize=10, first_n=1)

    layer4_7_fused = tf.add(layer4_1x1, layer7_upsampled)
    # layer4_7_fused = tf.Print(layer4_7_fused, [tf.shape(layer4_7_fused)], message= "Shape of layer4_7_fused:", summarize=10, first_n=1)

    layer4_7_upsampled = upsample_layer(layer4_7_fused, num_classes, "layer4_7_upsampled", 2)
    # layer4_7_upsampled = tf.Print(layer4_7_upsampled, [tf.shape(layer4_7_upsampled)], message= "Shape of layer4_7_upsampled:", summarize=10, first_n=1)

    layer3_1x1 = conv_1x1(vgg_layer3_out, num_classes)
    # layer3_1x1 = tf.Print(layer3_1x1, [tf.shape(layer3_1x1)], message= "Shape of layer3_1x1:", summarize=10, first_n=1)

    layer3_4_7_fused = tf.add(layer3_1x1, layer4_7_upsampled)
    # layer3_4_7_fused = tf.Print(layer3_4_7_fused, [tf.shape(layer3_4_7_fused)], message= "Shape of layer3_4_7_fused:", summarize=10, first_n=1)

    layer3_4_7_upsampled = upsample_layer(layer3_4_7_fused, num_classes, "layer3_4_7_upsampled", 8)
    # layer3_4_7_upsampled = tf.Print(layer3_4_7_upsampled, [tf.shape(layer3_4_7_upsampled)], message= "Shape of layer3_4_7_upsampled:", summarize=10, first_n=1)

    return layer3_4_7_upsampled

# tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # DONE: Implement function

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return logits, train_op, loss


# tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, batches_count):
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
    :param batches_count: number of batches in one epoch, used to correctly display progress
    """
    # DONE: Implement function
    loss_log = []
    for epoch in range(epochs):
        with tqdm(total=batches_count, unit=' batches') as progress_indicator:
            for image, ground_truth in get_batches_fn(batch_size):
                _, loss = sess.run([train_op, cross_entropy_loss],
                                   feed_dict={input_image: image, correct_label: ground_truth, keep_prob: 0.8,
                                              learning_rate: .00005})
                progress_indicator.set_description('Epoch {:>2}/{} - Loss {:.4f}'.format(epoch + 1, epochs, loss))
                progress_indicator.update(1)
        loss_log.append(loss)

    print()  # Go to new line after the last progress report

    return loss_log


# tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    epochs = 2
    batch_size = 2
    data_dir = './data'
    runs_dir = './runs'
    log_dir = './tensorboard'

    # Download pre-trained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # DONE: Build NN using load_vgg, layers, and optimize function
        vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor = load_vgg(
            sess=sess, vgg_path=vgg_path)

        output_layer = layers(vgg_layer3_out=vgg_layer3_out_tensor,
                              vgg_layer4_out=vgg_layer4_out_tensor,
                              vgg_layer7_out=vgg_layer7_out_tensor,
                              num_classes=num_classes)


        learning_rate = tf.placeholder(dtype=tf.float32)
        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))
        logits, train_op, loss = optimize(output_layer, correct_label, learning_rate, num_classes)

        # DONE: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        n_batches = helper.count_batches(os.path.join(data_dir, 'data_road/training'), batch_size)
        loss_log = train_nn(sess=sess,
                            epochs=epochs,
                            batch_size=batch_size,
                            get_batches_fn=get_batches_fn,
                            train_op=train_op,
                            cross_entropy_loss=loss,
                            input_image=vgg_input_tensor,
                            correct_label=correct_label,
                            keep_prob=vgg_keep_prob_tensor,
                            learning_rate=learning_rate,
                            batches_count=n_batches)

        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir=log_dir, graph=tf.get_default_graph())


        # DONE: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob_tensor,
                                      vgg_input_tensor)

        # Chart loss per epoch
        _, axes = plt.subplots()
        plt.plot(range(1, len(loss_log) + 1), loss_log)
        plt.title('Cross-entropy loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.grid()
        axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.show()


if __name__ == '__main__':
    run()
