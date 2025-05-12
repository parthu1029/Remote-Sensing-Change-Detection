
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, Dropout, concatenate, UpSampling2D, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from models.attention import cbam_block 


def build_unet_with_resnet(input_shape=(256, 256, 6), l2_lambda=0.01):
    inputs = Input(input_shape)
    input_image1 = inputs[..., :3]
    input_image2 = inputs[..., 3:]

    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(256, 256, 3))
    encoder_layers1 = [
        base_model.get_layer("conv1_relu").output,
        base_model.get_layer("conv2_block3_out").output,
        base_model.get_layer("conv3_block4_out").output,
        base_model.get_layer("conv4_block6_out").output,
        base_model.get_layer("conv5_block3_out").output
    ]
    encoder1 = Model(inputs=base_model.input, outputs=encoder_layers1)
    encoder1.trainable = False
    encoder_outputs1 = encoder1(input_image1)
    encoder_outputs2 = encoder1(input_image2)
    combined_features = [
        concatenate([f1, f2], axis=-1) for f1, f2 in zip(encoder_outputs1, encoder_outputs2)
    ]
    c1, c2, c3, c4, c5 = combined_features

    def conv_block(x, filters):
        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        return x

    # Decoder + CBAM
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, 512)
    c6 = cbam_block(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, 256)
    c7 = cbam_block(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, 128)
    c8 = cbam_block(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, 64)
    c9 = cbam_block(c9)

    u10 = UpSampling2D((2, 2))(c9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u10)

    model = Model(inputs, outputs)
    return model