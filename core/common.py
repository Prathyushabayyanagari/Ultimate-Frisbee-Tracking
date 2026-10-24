#! /usr/bin/env python

import tensorflow as tf


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Custom BatchNormalization class to handle "frozen state" and "inference mode" separately.
    When layer.trainable = False, the layer uses stored moving `mean` and `var` in inference mode.
    Additionally, `gamma` and `beta` are not updated when frozen.
    """

    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(
            training, self.trainable
        )  # Training also depends on layer's trainable status.
        return super().call(x, training)


def convolutional(
    input_layer,
    filters_shape,
    downsample=False,
    activate=True,
    bn=True,
    activate_type="leaky",
):
    """
    Implements a convolutional layer with options for downsampling, activation, and batch normalization.

    Args:
        input_layer: The input tensor.
        filters_shape: Tuple specifying kernel size and input/output channels (e.g., (3, 3, in_channels, out_channels)).
        downsample: If True, applies stride of 2 with zero padding for downsampling.
        activate: Whether to apply an activation function.
        bn: Whether to use batch normalization.
        activate_type: Type of activation function. Options: "leaky" (default) or "mish".

    Returns:
        The output tensor after applying convolution, optional batch normalization, and activation.
    """
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = "valid"
        strides = 2
    else:
        strides = 1
        padding = "same"

    # Convolution operation
    conv = tf.keras.layers.Conv2D(
        filters=filters_shape[-1],
        kernel_size=filters_shape[0],
        strides=strides,
        padding=padding,
        use_bias=not bn,
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.0),
    )(input_layer)

    # Optional batch normalization
    if bn:
        conv = BatchNormalization()(conv)

    # Optional activation function
    if activate:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
    return conv


def mish(x):
    """
    Implements the Mish activation function.
    Formula: x * tanh(softplus(x)), where softplus(x) = log(1 + exp(x)).
    """
    return x * tf.math.tanh(tf.math.softplus(x))


def residual_block(
    input_layer, input_channel, filter_num1, filter_num2, activate_type="leaky"
):
    """
    Implements a residual block with two convolutional layers.

    Args:
        input_layer: The input tensor.
        input_channel: Number of input channels.
        filter_num1: Number of filters in the first convolution.
        filter_num2: Number of filters in the second convolution.
        activate_type: Type of activation function. Options: "leaky" or "mish".

    Returns:
        The output tensor after applying the residual block.
    """
    short_cut = input_layer  # Shortcut connection
    conv = convolutional(
        input_layer,
        filters_shape=(1, 1, input_channel, filter_num1),
        activate_type=activate_type,
    )
    conv = convolutional(
        conv,
        filters_shape=(3, 3, filter_num1, filter_num2),
        activate_type=activate_type,
    )
    residual_output = short_cut + conv  # Element-wise addition
    return residual_output


def route_group(input_layer, groups, group_id):
    """
    Splits the input tensor into groups along the channel axis and returns a specific group.

    Args:
        input_layer: The input tensor.
        groups: Total number of groups to split into.
        group_id: The ID of the group to return.

    Returns:
        The tensor corresponding to the specified group.
    """
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]


def upsample(input_layer):
    """
    Upsamples the input tensor using bilinear interpolation.

    Args:
        input_layer: The input tensor.

    Returns:
        The upsampled tensor with double the height and width of the input.
    """
    return tf.image.resize(
        input_layer,
        (input_layer.shape[1] * 2, input_layer.shape[2] * 2),
        method="bilinear",
    )
