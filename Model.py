import keras

NUM_CLASSES = 5

def configure(num_classes=None):
    """
    Configure the parameters for this module if different from the default ones
    """
    global NUM_CLASSES, BATCH_SIZE
    if num_classes is not None:
        NUM_CLASSES = num_classes



def u_net(
          input_shape=(64,128,1), 
          depth=4, 
          d_conv_count = 3, 
          b_conv_count = 3, 
          u_conv_count = 3, 
          start_filter=32, 
          skip_connections=True, 
          dropout=0.2
          ):
    
    """
    Parametrizable definition of a UNet model, things to be tried and tuned are
    - Deeper network
    - Remove batch normalisation in upsampling path
    - Conv2DTranspose instead of bilinear interpolation for upsampling
    """

    print()
    print("FUNCTION U-NET")
    print("Building the model architecture...")
    # Input Layer
    inputs = keras.layers.Input(shape=input_shape, name='input_layer')

    normalized_inputs = keras.layers.Rescaling(1.0 / 255.0)(inputs)  # Normalize to [0, 1]

    x = normalized_inputs  # Pass normalized inputs to the rest of the model

    skipped = []
    # Downsampling
    for i in range(depth):
        for j in range(d_conv_count):
            x = keras.layers.Conv2D(filters=start_filter,
                            kernel_size=(3,3),
                            strides=(1, 1),
                            padding='same')(x)
            x = keras.layers.BatchNormalization()(x) 
            x = keras.layers.ReLU()(x)

        if skip_connections:
            # Save the layer for skip connections
            skipped.append(x)

        x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        x = keras.layers.Dropout(dropout)(x)
        
        start_filter = start_filter * 2
    
    # Bottleneck
    for j in range(b_conv_count):
        x = keras.layers.Conv2D(filters=start_filter,
                            kernel_size=(3,3),
                            strides=(1, 1),
                            padding='same')(x)
        x = keras.layers.BatchNormalization()(x) 
        x = keras.layers.ReLU()(x)
    
    start_filter = start_filter // 2

    # Upsampling
    for i in range(depth):
        x = keras.layers.UpSampling2D(2, interpolation='bilinear')(x)

        if skip_connections:
            x = keras.layers.Concatenate()([x, skipped[depth - i - 1]])

        x = keras.layers.Dropout(dropout)(x)

        for j in range(u_conv_count):
            x = keras.layers.Conv2D(filters=start_filter,
                            kernel_size=(3,3),
                            strides=(1, 1),
                            padding='same')(x)
            x = keras.layers.BatchNormalization()(x) ## remove?
            x = keras.layers.ReLU()(x)
        start_filter = start_filter // 2
    

    # Output Layer
    outputs = keras.layers.Conv2D(filters=NUM_CLASSES,
                        kernel_size=(1,1),
                        strides=(1, 1),
                        padding='same',
                        activation='softmax',
                        name="output_layer")(x)
    
    model = keras.Model(inputs, outputs, name='UNet')

    print("Completed model architecture")
    print()

    return model