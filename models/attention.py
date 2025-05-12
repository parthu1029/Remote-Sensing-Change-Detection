from tensorflow.keras.layers import ( Conv2D, Activation, Multiply, Dense, 
                                        GlobalAveragePooling2D, GlobalMaxPooling2D, 
                                        Reshape, Add, Conv2D, Concatenate, Lambda)

# CBAM block (Channel + Spatial Attention)
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_dense_one = Dense(channel // ratio, activation='relu', use_bias=False)
    shared_dense_two = Dense(channel, use_bias=False)

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    attention = Add()([avg_pool, max_pool])
    attention = Activation('sigmoid')(attention)

    return Multiply()([input_feature, attention])

def spatial_attention(input_feature):
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    attention = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    return Multiply()([input_feature, attention])

def cbam_block(input_feature):
    x = channel_attention(input_feature)
    x = spatial_attention(x)
    return x