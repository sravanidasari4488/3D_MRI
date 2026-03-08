"""
2D U-Net model definition for multi-class brain tumor segmentation.

Architecture:
- Input:  (240, 240, 4)  → 4 MRI modalities / channels
- Output: (240, 240, 3)  → 3-class segmentation with softmax activation
- Encoder: Conv2D + BatchNorm + ReLU × 2 → MaxPooling2D (4 levels)
- Bottleneck: Conv2D + BatchNorm + ReLU × 2
- Decoder: UpSampling2D → Concatenate skip connection → Conv2D + BN + ReLU × 2

Loss:
- Combined Dice loss (multi-class, averaged over classes)
  plus categorical cross-entropy.

Metric:
- Multi-class Dice coefficient.

This module only builds and compiles the model; it does not perform training.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K


def conv_block_2d(
    x: tf.Tensor,
    filters: int,
    kernel_size: Tuple[int, int] = (3, 3),
    padding: str = "same",
    activation: str = "relu",
) -> tf.Tensor:
    """
    Two consecutive Conv2D → BatchNorm → Activation blocks.
    """
    x = layers.Conv2D(
        filters,
        kernel_size,
        padding=padding,
        kernel_initializer="he_normal",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding=padding,
        kernel_initializer="he_normal",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    return x


def dice_coefficient_multiclass(
    y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6
) -> tf.Tensor:
    """
    Multi-class Dice coefficient.

    Expects y_true and y_pred with shape:
        (batch, H, W, C)
    where C is the number of classes.

    Computes Dice per class and returns the mean over classes.
    """
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")

    # Flatten over batch and spatial dimensions, keep classes separate.
    axes = (0, 1, 2)
    intersection = K.sum(y_true * y_pred, axis=axes)
    denominator = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)

    dice_per_class = (2.0 * intersection + smooth) / (denominator + smooth)
    return K.mean(dice_per_class)


def build_unet_2d(
    input_shape: Tuple[int, int, int] = (240, 240, 4),
    base_filters: int = 32,
    num_classes: int = 3,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """
    Build and compile a 2D U-Net model.

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape, default (240, 240, 4).
    base_filters : int
        Number of filters in the first encoder level (doubles each level).
    num_classes : int
        Number of output channels/classes (default: 3).
    learning_rate : float
        Learning rate for the Adam optimizer.

    Returns
    -------
    model : tf.keras.Model
        Compiled 2D U-Net model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Encoder
    c1 = conv_block_2d(inputs, filters=base_filters)
    p1 = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(c1)

    c2 = conv_block_2d(p1, filters=base_filters * 2)
    p2 = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(c2)

    c3 = conv_block_2d(p2, filters=base_filters * 4)
    p3 = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(c3)

    c4 = conv_block_2d(p3, filters=base_filters * 8)
    p4 = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(c4)

    # Bottleneck
    bn = conv_block_2d(p4, filters=base_filters * 16)

    # Decoder
    u4 = layers.UpSampling2D(size=(2, 2), name="up4")(bn)
    u4 = layers.Concatenate(name="concat4")([u4, c4])
    c5 = conv_block_2d(u4, filters=base_filters * 8)

    u3 = layers.UpSampling2D(size=(2, 2), name="up3")(c5)
    u3 = layers.Concatenate(name="concat3")([u3, c3])
    c6 = conv_block_2d(u3, filters=base_filters * 4)

    u2 = layers.UpSampling2D(size=(2, 2), name="up2")(c6)
    u2 = layers.Concatenate(name="concat2")([u2, c2])
    c7 = conv_block_2d(u2, filters=base_filters * 2)

    u1 = layers.UpSampling2D(size=(2, 2), name="up1")(c7)
    u1 = layers.Concatenate(name="concat1")([u1, c1])
    c8 = conv_block_2d(u1, filters=base_filters)

    # Output layer: num_classes channels with softmax activation.
    outputs = layers.Conv2D(
        filters=num_classes,
        kernel_size=(1, 1),
        activation="softmax",
        padding="same",
        name="output_mask",
    )(c8)

    model = models.Model(inputs=inputs, outputs=outputs, name="UNet2D")

    # Loss: Dice loss + categorical cross-entropy.
    cce = tf.keras.losses.CategoricalCrossentropy()

    def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return 1.0 - dice_coefficient_multiclass(y_true, y_pred)

    def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return cce(y_true, y_pred) + dice_loss(y_true, y_pred)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[dice_coefficient_multiclass],
    )

    return model

