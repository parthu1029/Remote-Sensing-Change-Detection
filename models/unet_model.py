from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, concatenate,
                                     Conv2DTranspose, BatchNormalization, Activation)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from models.attention import cbam_block 

def conv_block(x, filters, dropout_rate=0.3, l2_lambda=0.01):
    x = Conv2D(filters, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dropout(dropout_rate)(x)
    x = cbam_block(x)
    return x

def build_unet(input_size=(256, 256, 6), l2_lambda=0.001):
    inputs = Input(input_size)
    
    input_image1 = inputs[..., :3]
    input_image2 = inputs[..., 3:]

    # Encoder Image 1
    c1_1 = conv_block(input_image1, 32, dropout_rate=0.1, l2_lambda=l2_lambda)
    p1_1 = MaxPooling2D((2, 2))(c1_1)

    c2_1 = conv_block(p1_1, 64, dropout_rate=0.1, l2_lambda=l2_lambda)
    p2_1 = MaxPooling2D((2, 2))(c2_1)

    c3_1 = conv_block(p2_1, 128, dropout_rate=0.2, l2_lambda=l2_lambda)
    p3_1 = MaxPooling2D((2, 2))(c3_1)

    c4_1 = conv_block(p3_1, 256, dropout_rate=0.3, l2_lambda=l2_lambda)
    p4_1 = MaxPooling2D((2, 2))(c4_1)

    # Encoder Image 2
    c1_2 = conv_block(input_image2, 32, dropout_rate=0.1, l2_lambda=l2_lambda)
    p1_2 = MaxPooling2D((2, 2))(c1_2)

    c2_2 = conv_block(p1_2, 64, dropout_rate=0.1, l2_lambda=l2_lambda)
    p2_2 = MaxPooling2D((2, 2))(c2_2)

    c3_2 = conv_block(p2_2, 128, dropout_rate=0.2, l2_lambda=l2_lambda)
    p3_2 = MaxPooling2D((2, 2))(c3_2)

    c4_2 = conv_block(p3_2, 256, dropout_rate=0.3, l2_lambda=l2_lambda)
    p4_2 = MaxPooling2D((2, 2))(c4_2)

    # Bottleneck
    b1 = conv_block(p4_1, 256, dropout_rate=0.4, l2_lambda=l2_lambda)
    b2 = conv_block(p4_2, 256, dropout_rate=0.4, l2_lambda=l2_lambda)
    bottleneck = concatenate([b1, b2])

    # Decoder
    u1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bottleneck)
    u1 = concatenate([u1, c4_1, c4_2])
    d1 = conv_block(u1, 256, dropout_rate=0.3, l2_lambda=l2_lambda)

    u2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(d1)
    u2 = concatenate([u2, c3_1, c3_2])
    d2 = conv_block(u2, 128, dropout_rate=0.2, l2_lambda=l2_lambda)

    u3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d2)
    u3 = concatenate([u3, c2_1, c2_2])
    d3 = conv_block(u3, 64, dropout_rate=0.1, l2_lambda=l2_lambda)

    u4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(d3)
    u4 = concatenate([u4, c1_1, c1_2])
    d4 = conv_block(u4, 32, dropout_rate=0.1, l2_lambda=l2_lambda)

    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(d4)
    model = Model(inputs=inputs, outputs=outputs)

    return model
