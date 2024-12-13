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
                            padding='same',
                            kernel_initializer="he_normal")(x)
            
            x = keras.layers.Dropout(dropout)(x)
            x = keras.layers.BatchNormalization()(x) 
            x = keras.layers.ReLU()(x)

        if skip_connections:
            # Save the layer for skip connections
            skipped.append(x)

        x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        
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
        x = keras.layers.Conv2DTranspose(
                filters=start_filter,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="same",
                )(x)

        if skip_connections:
            x = keras.layers.Concatenate()([x, skipped[depth - i - 1]])

        x = keras.layers.Dropout(dropout)(x)

        for j in range(u_conv_count):
            x = keras.layers.Conv2D(filters=start_filter,
                            kernel_size=(3,3),
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer="he_normal")(x)
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




#############################
# ATTENTION UNET            #
#############################

def attention_block(x, g, inter_channel):
    theta_x = keras.layers.Conv2D(inter_channel, (1, 1), padding="same")(x)  # Query
    phi_g = keras.layers.Conv2D(inter_channel, (1, 1), padding="same")(g)  # Key
    attn = keras.layers.Activation("softmax")(
        keras.layers.Add()([theta_x, phi_g])
    )  # Attention Map
    attn = keras.layers.Conv2D(1, (1, 1), padding="same")(attn)  # Value
    return keras.layers.Multiply()([x, attn])


def attention_u_net(
    input_shape=(64,128,1),
    depth=4,
    d_conv_count=3,
    b_conv_count=3,
    u_conv_count=3,
    start_filter=32,
    skip_connections=True,
    dropout=0.2,
    NUM_CLASSES=5,
    ):

    print()
    print("FUNCTION ATTENTION U-NET")
    print("Building the model architecture...")

    # Input Layer
    inputs = keras.layers.Input(shape=input_shape, name="input_layer")

    normalized_inputs = keras.layers.Rescaling(1.0 / 255.0)(inputs)  # Normalize to [0, 1]

    x = normalized_inputs  # Pass normalized inputs to the rest of the model
    
    skipped = []

    # Downsampling
    for i in range(depth):

        for j in range(d_conv_count):
            x = keras.layers.Conv2D(
                filters=start_filter, kernel_size=(3, 3), strides=(1, 1), padding="same"
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)

        if skip_connections:
            # Save the layer for skip connections
            skipped.append(x)

        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Dropout(dropout)(x)

        start_filter = start_filter * 2

    # Bottleneck
    residual = x
    for j in range(b_conv_count):
        x = keras.layers.Conv2D(filters=start_filter, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

    # Match shapes for residual connection
    residual = keras.layers.Conv2D(
        filters=start_filter, kernel_size=(1, 1), padding="same"
    )(residual)
    x = keras.layers.Add()([x, residual])  # Residual Connection


    start_filter = start_filter // 2

    # Upsampling
    for i in range(depth):

        x = keras.layers.Conv2DTranspose(start_filter, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
        
        if skip_connections:
            # Apply attention mechanism to the skip connections
            x = attention_block(x, skipped[depth - i - 1], start_filter)

        x = keras.layers.Dropout(dropout)(x)

        for j in range(u_conv_count):
            x = keras.layers.Conv2D(
                filters=start_filter, kernel_size=(3, 3), strides=(1, 1), padding="same"
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)

        start_filter = start_filter // 2

    # Output Layer
    outputs = keras.layers.Conv2D(
        filters=NUM_CLASSES,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        activation="softmax",
        name="output_layer",
    )(x)

    model = keras.Model(inputs, outputs, name="Attention_UNet")
    return model


#############################
# DENSE UNET                #
#############################

def dense_u_net(
          input_shape=(64,128,1), 
          depth=4,
          d_conv_count = 3, 
          b_conv_count = 3, 
          u_conv_count = 3, 
          start_filter=32, 
          dropout=0.2
          ):
    
    print()
    print("FUNCTION DENSE U-NET")
    print("Building the model architecture...")
    # Input Layer
    inputs = keras.layers.Input(shape=input_shape, name='input_layer')

    normalized_inputs = keras.layers.Rescaling(1.0 / 255.0)(inputs)  # Normalize to [0, 1]

    next = normalized_inputs  # Pass normalized inputs to the rest of the model

    skipped = []
    growth_rate = 16
    # Downsampling
    for i in range(depth):
        for j in range(d_conv_count): # Dense Block
            x = keras.layers.Conv2D(filters=start_filter,
                            kernel_size=(3,3),
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer="he_normal")(next)
            
            x = keras.layers.BatchNormalization()(x) 
            conv_block = keras.layers.ReLU()(x)
            
            next = keras.layers.Concatenate()([conv_block, next])

        skipped.append(conv_block)

        x = keras.layers.MaxPooling2D(pool_size=(2,2))(conv_block)
        next = keras.layers.Dropout(dropout)(x)
        
        start_filter = start_filter * 2
    
    # Bottleneck
    for j in range(b_conv_count):
        x = keras.layers.Conv2D(filters=start_filter,
                            kernel_size=(3,3),
                            strides=(1, 1),
                            kernel_initializer="he_normal",
                            padding='same')(next)
        x = keras.layers.BatchNormalization()(x) 
        x = keras.layers.ReLU()(x)
        next = keras.layers.Dropout(dropout)(x)

    start_filter = start_filter // 2

    # Upsampling
    for i in range(depth):
        x = keras.layers.Conv2DTranspose(
                filters=start_filter,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="same",
                )(next)
        x = keras.layers.Concatenate()([x, skipped.pop()])
        next = keras.layers.Dropout(dropout)(x)

        for j in range(u_conv_count): # Dense Block
            x = keras.layers.Conv2D(filters=start_filter,
                            kernel_size=(3,3),
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer="he_normal")(next)            
            x = keras.layers.BatchNormalization()(x)
            conv_block = keras.layers.ReLU()(x)
            next =  keras.layers.Concatenate()([conv_block, next])
        start_filter = start_filter // 2
    
    # Output Layer
    outputs = keras.layers.Conv2D(filters=NUM_CLASSES,
                        kernel_size=(1,1),
                        strides=(1, 1),
                        padding='same',
                        activation='softmax',
                        name="output_layer")(next)
    
    model = keras.Model(inputs, outputs, name='Dense_UNet')

    print("Completed model architecture")
    print()

    return model
