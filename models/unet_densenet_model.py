from tensorflow.keras import Input, Model
from tensorflow.keras.layers import UpSampling2D, concatenate, Conv2D, BatchNormalization, Activation, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.applications import DenseNet121
from models.attention import cbam_block 

# Decoder block with BN + CBAM
def decoder_block(x, skip, filters, l2_lambda):
    x = UpSampling2D((2, 2))(x)
    x = concatenate([x, skip])

    x = Conv2D(filters, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Conv2D(filters, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = cbam_block(x)
    return x
    
def build_unet_with_densenet(input_shape=(256, 256, 6), l2_lambda=0.01):
    inputs = Input(input_shape)
    input_image1 = inputs[..., :3]
    input_image2 = inputs[..., 3:]

    base_model = DenseNet121(include_top=False, weights="imagenet", input_shape=(256, 256, 3))

    encoder_layers1 = [
        base_model.get_layer("conv1_relu").output, 
        base_model.get_layer("pool2_conv").output, 
        base_model.get_layer("pool3_conv").output, 
        base_model.get_layer("pool4_conv").output, 
        base_model.get_layer("conv5_block16_concat").output
    ]

    encoder = Model(inputs=base_model.input, outputs=encoder_layers1)
    encoder.trainable = False

    # Apply encoder to both images
    encoder_outputs1 = encoder(input_image1)
    encoder_outputs2 = encoder(input_image2)

    # Concatenate corresponding feature maps
    combined_features = [
        concatenate([e1, e2], axis=-1, name=f"concat_{i}")
        for i, (e1, e2) in enumerate(zip(encoder_outputs1, encoder_outputs2))
    ]

    c1, c2, c3, c4, c5 = combined_features

    # Decoder
    c6 = decoder_block(c5, c4, 512, l2_lambda)
    c7 = decoder_block(c6, c3, 256, l2_lambda)
    c8 = decoder_block(c7, c2, 128, l2_lambda)
    c9 = decoder_block(c8, c1, 64, l2_lambda)

    u10 = UpSampling2D((2, 2))(c9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u10)

    return Model(inputs=inputs, outputs=outputs)