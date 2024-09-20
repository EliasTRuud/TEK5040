import tensorflow as tf
from tensorflow.keras import layers, models

# Theory
# https://www.jeremyjordan.me/semantic-segmentation/ 

def simple_model(input_shape):

    height, width, channels = input_shape
    image = layers.Input(input_shape)
    x = layers.Conv2D(32, 5, strides=(2, 2), padding='same', activation='relu')(image)
    x = layers.Conv2D(64, 5, strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(1, 1, padding='same', activation=None)(x)
    # resize back into same size as regularization mask
    x = tf.image.resize(x, [height, width])
    x = tf.keras.activations.sigmoid(x)

    model = models.Model(inputs=image, outputs=x)

    return model


def conv2d_3x3(filters):
    conv = layers.Conv2D(
        filters, kernel_size=(3, 3), activation='relu', padding='same'
    )
    return conv


def max_pool():
    return layers.MaxPooling2D((2, 2), strides=2, padding='same')

def conv2d_3x3_Up(filters):
    # Similar to downsampling with conv2d_3x3, but instead we upsample
    # Kernel size and activation 2x2 from figure. Set strides to 2,2 such dimensions will match
    conv = layers.Conv2DTranspose(
            filters, kernel_size=(2, 2), strides=(2,2), activation='relu', padding='same'
        )
    return conv

def unet(input_shape):
    '''
    Unet model which downsamples then upsamples again with skip connections between layers on same level
    https://www.geeksforgeeks.org/u-net-architecture-explained/
    At the bottleneck, a final convolution operation is performed to generate a 30 30 1024 (our case 128) shaped 
    feature map. The expansive path then takes the feature map from the bottleneck and converts it 
    back into an image of the same size as the original input.
    '''

    image = layers.Input(shape=input_shape)
    # Fill the layers from 2 to 9.
    # 8 -> 16 -> 32 -> 64 -> 128 then back up again -> 64 -> 32 -> 16 -> 8
    
    # Contradicting path
    c1 = conv2d_3x3(8)(image)
    c1 = conv2d_3x3(8)(c1)
    p1 = max_pool()(c1)

    c2 = conv2d_3x3(16)(p1)
    c2 = conv2d_3x3(16)(c2)
    p2 = max_pool()(c2)

    c3 = conv2d_3x3(32)(p2)
    c3 = conv2d_3x3(32)(c3)
    p3 = max_pool()(c3)

    c4 = conv2d_3x3(64)(p3)
    c4 = conv2d_3x3(64)(c4)
    p4 = max_pool()(c4)

    #Bottleneck
    c5 = conv2d_3x3(128)(p4)
    c5 = conv2d_3x3(128)(c5)

    # Expansive path
    # Upsample -> copy and group -> conv2d_3x3 two times.
    # Used as example: https://github.com/Nguyendat-bit/U-net/blob/main/model.py
    u6 = conv2d_3x3_Up(64)(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = conv2d_3x3(64)(u6)
    c6 = conv2d_3x3(64)(c6)

    u7 = conv2d_3x3_Up(32)(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = conv2d_3x3(32)(u7)
    c7 = conv2d_3x3(32)(c7)

    u8 = conv2d_3x3_Up(16)(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = conv2d_3x3(16)(u8)
    c8 = conv2d_3x3(16)(c8)

    u9 = conv2d_3x3_Up(8)(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = conv2d_3x3(8)(u9)
    c9 = conv2d_3x3(8)(c9)

    probs = layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(c9)
    model = models.Model(inputs=image, outputs=probs)
    
    return model
