from keras.models import Model
from keras.layers import Input, Conv1D, Conv3D, BatchNormalization, Activation, concatenate, Add, Dense, \
    GlobalAveragePooling3D


def residual_block(x,y, filters, kernel_size=3):
    # Shortcut
    shortcut = x

    # 1D convolution
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 3D convolution
    y = Conv3D(filters, (1, kernel_size, kernel_size), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    # Residual connection
    x = Add()([x, y])

    return x


def dense_block(x, y,filters, num_layers=4):
    for _ in range(num_layers):
        x = residual_block(x,y, filters)
    return x


def hybrid_residual_densenet(sol,input_shape1=(32, 32, 32, 1),input_shape2=(32, 32, 32, 1), num_classes=10):
    input_data1 = Input(shape=input_shape1)
    input_data2 = Input(shape=input_shape2)
    x = Conv1D(sol[0], 3, padding='same')(input_data1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Initial 3D convolution
    y = Conv3D(sol[0], (1, 3, 3), padding='same')(input_data2)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    data = x
    # Hybrid residual dense blocks
    for num_filters in [64, 128, 256]:
        dense_output = dense_block(x,y, num_filters)
        data = concatenate([x,y, dense_output], axis=-1)

    # Global Average Pooling and Dense layers
    x = GlobalAveragePooling3D()(data)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_data1, outputs=x)

    return model,data

def Model_HCARDNet(Feature,Video,Target,sol=None):
    if sol is None:
        sol = [5,5,100]
    # Create the model
    model,Data= hybrid_residual_densenet(sol,Feature,Video)
    Activation_function = round(Data.shape[0] * 0.75)
    data = Data[:Activation_function, :]
    labels = Target[:Activation_function, :]
    test_data = Data[Activation_function:, :]
    test_target = Target[Activation_function:, :]
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(data, labels.astype('int'), epochs=sol[1], Batch_size = sol[2],validation_data=(test_data, test_target.astype('int')))
    pred = model.predict(test_data)
    return pred

