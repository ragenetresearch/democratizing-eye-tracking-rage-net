from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Multiply

from Models.ResNet.ResNet18Model import ResNet18Model


def create_resnet18_sw__attention(input_shape: Tuple = (60, 36, 1),
                                  bn: bool = False,
                                  first_dense_units: int = 256,
                                  fc_layer_units: list = None,
                                  debug=True) -> Model:
    fc_layer_units = fc_layer_units if fc_layer_units is not None else [256, 512]
    if debug:
        print('!Note: Using model ResNet18Model with shared weights between the two eyes')
    cnn_feature_extractor = ResNet18Model()
    cnn_attention_extractor = ResNet18Model()

    # CNN part for right eye
    eye_right_input = Input(input_shape)
    eye_right_out = cnn_feature_extractor(eye_right_input)
    eye_right_out = Dense(units=first_dense_units, activation='relu')(eye_right_out)
    eye_right_out = BatchNormalization()(eye_right_out)

    eye_right_attention = cnn_attention_extractor(eye_right_input)
    eye_right_attention = Dense(units=first_dense_units, activation='sigmoid')(eye_right_attention)

    eye_right_out = Multiply()([eye_right_out, eye_right_attention])

    # CNN part for left eye
    eye_left_input = Input(input_shape)
    eye_left_out = cnn_feature_extractor(eye_left_input)
    eye_left_out = Dense(units=first_dense_units, activation='relu')(eye_left_out)
    eye_left_out = BatchNormalization()(eye_left_out)

    eye_left_attention = cnn_attention_extractor(eye_left_input)
    eye_left_attention = Dense(units=first_dense_units, activation='sigmoid')(eye_left_attention)

    eye_left_out = Multiply()([eye_left_out, eye_left_attention])

    # Concatenate multiple heads
    output = tf.concat([eye_right_out, eye_left_out], axis=1)

    # Fully connected layers
    for layer_units in fc_layer_units:
        output = Dense(units=layer_units, activation='relu')(output)
        if bn:
            output = BatchNormalization()(output)

    output = Dense(units=2, activation='sigmoid')(output)

    # Create the Model
    return Model(inputs=[eye_right_input, eye_left_input], outputs=output)
