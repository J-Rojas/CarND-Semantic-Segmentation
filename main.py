import os.path
import tensorflow as tf
import helper
import warnings
import typing
import sys
from distutils.version import LooseVersion
import project_tests as tests

FREEZE_VGG = False

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def print_variables(sess, variables_names):
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print ("Variable: ", k)
        print ("Shape: ", v.shape)

def trainable_variables():
    trainable_variables = []
    if FREEZE_VGG:
        # get trainable variables from 'fcn' scope
        trainable_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fcn')

    if len(trainable_variables) == 0:
        trainable_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    return trainable_variables

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    vgg_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3 = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4 = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7 = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

    return \
    vgg_input, \
    vgg_keep_prob, \
    vgg_layer3, \
    vgg_layer4, \
    vgg_layer7

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

    # add 1x1 convolutional layers
    #vgg_layer3_out = tf.Print(vgg_layer3_out, ("vgg_layer3_out shape ", tf.shape_n([vgg_layer3_out])[0][1:4]))
    #vgg_layer4_out = tf.Print(vgg_layer4_out, ("vgg_layer4_out shape ", tf.shape_n([vgg_layer4_out])[0][1:4]))
    #vgg_layer7_out = tf.Print(vgg_layer7_out, ("vgg_layer7_out shape ", tf.shape_n([vgg_layer7_out])[0][1:4]))

    regularizer = tf.contrib.layers.l2_regularizer(1e-3)
    activation = tf.nn.relu
    initializer = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope("fcn"):
        fc_layer_1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1),
                                      padding='same',
                                      kernel_initializer=initializer,
                                      kernel_regularizer=regularizer,
                                      #activation=activation,
                                      name='fc_layer_1')
        #fc_layer_1 = tf.Print(fc_layer_1, ("fc_layer_1 shape ", tf.shape_n([fc_layer_1])[0][1:4]))

        fc_layer_2 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1),
                                    padding='same',
                                    kernel_initializer=initializer,
                                    kernel_regularizer=regularizer,
                                    #activation=activation,
                                    name='fc_layer_2')
        #fc_layer_2 = tf.Print(fc_layer_2, ("fc_layer_2 shape ", tf.shape_n([fc_layer_2])[0][1:4]))

        fc_layer_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1),
                                    padding='same',
                                    kernel_initializer=initializer,
                                    kernel_regularizer=regularizer,
                                    #activation=activation,
                                    name='fc_layer_3')
        #fc_layer_3 = tf.Print(fc_layer_3, ("fc_layer_3 shape ", tf.shape_n([fc_layer_3])[0][1:4]))

        # add layers of deconvolutions for FCN-8 (3 upscaling layers of 2x each = 8x)

        # 2x upscale
        upscale_1 = tf.layers.conv2d_transpose(fc_layer_1, num_classes, 4, strides=(2, 2),
                                    kernel_initializer=initializer,
                                    kernel_regularizer=regularizer,
                                    padding='same',
                                    #activation=activation,
                                    name='upscale_1')
        #upscale_1 = tf.Print(upscale_1, ("upscale_1 shape ", tf.shape_n([upscale_1])[0][1:4]))
        #upscale_1 = tf.Print(upscale_1, ("vgg_layer4_out shape ", tf.shape_n([vgg_layer4_out])[0][1:4]))

        # skip connection with pooled layer 4
        skip_1 = tf.add(upscale_1, fc_layer_2, name='skip_1')

        #skip_1 = tf.Print(skip_1, ("skip_1 shape ", tf.shape_n([skip_1])[0][1:4]))

        # 2x upscale
        upscale_2 = tf.layers.conv2d_transpose(skip_1, num_classes, 4, strides=(2, 2),
                                    kernel_initializer=initializer,
                                    kernel_regularizer=regularizer,
                                    padding='same',
                                    #activation=activation,
                                    name='upscale_2')

        # skip connection with pooled layer 3
        skip_2 = tf.add(upscale_2, fc_layer_3, name='skip_2')

        # 4x upscale
        upscale_3 = tf.layers.conv2d_transpose(skip_2, num_classes, 8, strides=(4, 4),
                                    kernel_initializer=initializer,
                                    kernel_regularizer=regularizer,
                                    #activation=activation,
                                    padding='same',
                                    name='upscale_3')

        # convolution to correlate neighboring pixel losses with each other... better precision/less noise since neighboring pixels should be related
        conv_3_1 = tf.layers.conv2d(upscale_3, num_classes, 3, strides=(1, 1),
                                    kernel_initializer=initializer,
                                    kernel_regularizer=regularizer,
                                    activation=activation,
                                    padding='same',
                                    name='conv_3_1')

        # avg pool 2x downscale - similar reasoning as above
        downscale_3_2 = tf.layers.average_pooling2d(conv_3_1, pool_size=(3, 3), strides=(2, 2),
                                    padding='same',
                                    name='downscale_3_2')

        # 4x upsample to match original image
        upscale_4 = tf.layers.conv2d_transpose(downscale_3_2, num_classes, 8, strides=(4, 4),
                                    kernel_initializer=initializer,
                                    kernel_regularizer=regularizer,
                                    #activation=activation,
                                    padding='same',
                                    name='upscale_4')

        # convolution to correlate neighboring pixel losses with each other... better precision/less noise since neighboring pixels should be related
        conv_4_1 = tf.layers.conv2d(upscale_4, num_classes, 3, strides=(1, 1),
                                    kernel_initializer=initializer,
                                    kernel_regularizer=regularizer,
                                    activation=activation,
                                    padding='same',
                                    name='conv_4_1')

        # avg pool 2x downscale - similar reasoning as above
        downscale_4_2 = tf.layers.average_pooling2d(conv_4_1, pool_size=(3, 3), strides=(2, 2),
                                    padding='same',
                                    name='downscale_4_2')

        # 4x upsample to match original image
        upscale_5 = tf.layers.conv2d_transpose(downscale_4_2, num_classes, 8, strides=(4, 4),
                                    kernel_initializer=initializer,
                                    kernel_regularizer=regularizer,
                                    #activation=activation,
                                    padding='same',
                                    name='upscale_5')

        # convolution to correlate neighboring pixel losses with each other... better precision/less noise since neighboring pixels should be related
        conv_5_1 = tf.layers.conv2d(upscale_5, num_classes, 3, strides=(1, 1),
                                    kernel_initializer=initializer,
                                    kernel_regularizer=regularizer,
                                    activation=activation,
                                    padding='same',
                                    name='conv_5_1')

        # avg pool 2x downscale - similar reasoning as above
        downscale_5_2 = tf.layers.average_pooling2d(conv_5_1, pool_size=(3, 3), strides=(2, 2),
                                    padding='same',
                                    name='downscale_5_2')

        #upscale_3 = tf.Print(upscale_3, ("upscale_3 shape ", tf.shape_n([upscale_3])[0][1:4]))

    print(downscale_5_2.name)

    return downscale_5_2
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

    #print(trainable_variables)

    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="logits")
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label), name="cross_entropy_loss")
    trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = trainer.minimize(cross_entropy_loss, var_list=trainable_variables(), name="train_op")

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, saver=None):
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

    variables_names = [v.name for v in tf.trainable_variables()]
    print_variables(sess, variables_names)

    for epoch in range(epochs):
        total_loss = 0
        print('Epoch ', epoch)
        #retrieve training data
        batch = 1
        for train_data, label_data in get_batches_fn(batch_size):
            feed_dict={input_image: train_data, correct_label: label_data, learning_rate: 0.001, keep_prob: 0.9}
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict)
            #print('    batch ', batch, ' loss = ', loss)
            batch += 1
            total_loss += loss

        print('    avg loss = ', total_loss / (batch - 1))

        if saver:
            saver.save(sess, './runs/savedModel', global_step=epoch+1)
            print('    saving to ./runs/savedModel-{}'.format(epoch+1))

tests.test_train_nn(train_nn)

def run(modelFile):
    num_classes = 2
    epochs = 30
    batch_size = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    save_file = None
    if modelFile:
        save_file = './' + modelFile # String addition used for emphasis

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function

        # Load model
        if save_file:
            saver = tf.train.import_meta_graph(save_file + '.meta')
            saver.restore(sess, save_file)
            input_image, keep_prob_layer, vgg_layer_3, vgg_layer_4, vgg_layer_7 = \
                tf.get_default_graph().get_tensor_by_name("image_input:0"), \
                tf.get_default_graph().get_tensor_by_name("keep_prob:0"), \
                tf.get_default_graph().get_tensor_by_name("layer3_out:0"), \
                tf.get_default_graph().get_tensor_by_name("layer4_out:0"), \
                tf.get_default_graph().get_tensor_by_name("layer7_out:0")
            nn_last_layer = tf.get_default_graph().get_tensor_by_name("fcn/upscale_3/bias:0")
            logits = tf.get_default_graph().get_tensor_by_name("logits:0")
            cross_entropy_loss = tf.get_default_graph().get_tensor_by_name("cross_entropy_loss:0")
            correct_label = tf.get_default_graph().get_tensor_by_name("correct_label:0")
            learning_rate = tf.get_default_graph().get_tensor_by_name("learning_rate:0")
            train_op = tf.get_default_graph().get_operation_by_name("train_op")
            print('    loading layers from {}'.format(save_file))
        else:
            correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes], name='correct_label')
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            input_image, keep_prob_layer, vgg_layer_3, vgg_layer_4, vgg_layer_7 = load_vgg(sess, vgg_path)
            nn_last_layer = layers(vgg_layer_3, vgg_layer_4, vgg_layer_7, num_classes)
            logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)

        #tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #tf.summary.merge_all()
        #train_writer = tf.summary.FileWriter('./summary/train',
        #                              sess.graph)

        if save_file is None:
            saver.save(sess, './runs/savedModel', global_step=0)
            print('    saving to ./runs/savedModel-{}'.format(0))

        #sess.run(tf.variables_initializer())

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob_layer, learning_rate,
                 saver=saver)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob_layer, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    model_file = sys.argv[1] if len(sys.argv) > 1 else None
    if model_file:
        print('Loading model file {}'.format(model_file))
    run(model_file)
