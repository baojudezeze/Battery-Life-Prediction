from keras import layers
from tensorflow import keras


def build_model_101(input_shape):
    # 输入层:(100,109,109,3)
    inputs = keras.Input(shape=input_shape)

    # CBS(k=3,s=1):(100,109,109,3)-->(34,37,37,16)
    x = layers.Conv3D(16, 5, strides=3, activation='relu')(inputs)

    # (34, 37, 37, 16)-->(17,18,18,16)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (17,18,18,16)-->(5,5,5,32)
    x = layers.Conv3D(32, 5, strides=4, activation='relu')(x)

    # (5,5,5,32)-->(1,1,1,32)
    x = layers.MaxPooling3D((3, 4, 4))(x)

    # 展平层：(1,1,1,32)-->(32)
    x = layers.Flatten()(x)

    # 全连接层+输出层
    x = layers.Dense(units=16, activation='relu')(x)
    x = layers.Dense(units=4, activation='relu')(x)
    outputs = layers.Dense(units=1)(x)

    # 构建模型
    model = keras.Model(inputs, outputs)
    model.summary()
    return model
    # 可以考虑zeropadding3d层


def build_model_102(input_shape):
    # 输入层:(100,109,109,3)
    inputs = keras.Input(shape=input_shape)

    # CBS(k=3,s=1):(100,109,109,3)-->(34,37,37,16)
    x = layers.Conv3D(16, 5, strides=3, activation='relu')(inputs)

    # (34, 37, 37, 16)-->(17,18,18,16)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (17,18,18,16)-->(5,5,5,32)
    x = layers.Conv3D(32, 5, strides=4, activation='relu')(x)

    # (5,5,5,32)-->(1,1,1,32)
    x = layers.MaxPooling3D((3, 4, 4))(x)

    # 展平层：(1,1,1,32)-->(32)
    x = layers.Flatten()(x)

    # 全连接层+输出层
    x = layers.Dense(units=16, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=4, activation='relu')(x)
    outputs = layers.Dense(units=1)(x)

    # 构建模型
    model = keras.Model(inputs, outputs)
    model.summary()
    return model
    # 可以考虑zeropadding3d层


def build_model_103(input_shape):
    # 输入层:(100,109,109,3)
    inputs = keras.Input(shape=input_shape)

    # CBS(k=3,s=1):(100,109,109,3)-->(34,37,37,16)
    x = layers.Conv3D(16, 5, strides=3, activation='relu')(inputs)

    # (34, 37, 37, 16)-->(17,18,18,16)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (17,18,18,16)-->(5,5,5,32)
    x = layers.Conv3D(32, 5, strides=4, activation='relu')(x)

    # (5,5,5,32)-->(1,1,1,32)
    x = layers.MaxPooling3D((3, 4, 4))(x)

    # 展平层：(1,1,1,32)-->(32)
    x = layers.Flatten()(x)

    # 全连接层+输出层
    x = layers.Dense(units=16, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(units=4, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(units=1)(x)

    # 构建模型
    model = keras.Model(inputs, outputs)
    model.summary()
    return model
    # 可以考虑zeropadding3d层


def build_model_104(input_shape):
    # 输入层:(100,109,109,3)
    inputs = keras.Input(shape=input_shape)

    # CBS(k=3,s=1):(100,109,109,3)-->(50,55,55,16)
    x = layers.Conv3D(16, 5, strides=2, activation='relu', padding='same')(inputs)

    # (50,55,55,16)-->(25,28,28,16)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (25,28,28,16)-->(13,14,14,32)
    x = layers.Conv3D(32, 3, strides=2, activation='relu', padding='same')(x)

    # (13,14,14,32)-->(7,7,7,32)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (7,7,7,32) -->(4, 4, 4, 64)
    x = layers.Conv3D(64, 3, strides=2, activation='relu', padding='same')(x)

    # (4, 4, 4, 64)-->(1, 1, 1, 64)
    x = layers.MaxPooling3D((3, 3, 3))(x)

    # 展平层：(1,1,1,64)-->(64)
    x = layers.Flatten()(x)

    # 全连接层+输出层
    x = layers.Dense(units=32, activation='relu')(x)
    x = layers.Dropout(0.9)(x)
    x = layers.Dense(units=16, activation='relu')(x)
    x = layers.Dropout(0.9)(x)
    x = layers.Dense(units=4, activation='relu')(x)
    x = layers.Dropout(0.9)(x)
    outputs = layers.Dense(units=1, activation='relu')(x)

    # 构建模型
    model = keras.Model(inputs, outputs)
    model.summary()
    return model
    # 可以考虑zeropadding3d层


def build_model_105(input_shape, window):
    # 输入层:(100,109,109,3)
    inputs = keras.Input(shape=input_shape)

    # CBS(k=3,s=1):(100,109,109,3)-->(50,55,55,16)
    x = layers.Conv3D(16, 3, strides=2, activation='relu', padding='same')(inputs)

    # (50,55,55,16)-->(25,28,28,16)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (25,28,28,16)-->(13,14,14,32)
    x = layers.Conv3D(32, 3, strides=2, activation='relu', padding='same')(x)

    # (13,14,14,32)-->(7,7,7,32)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (7,7,7,32) -->(4, 4, 4, 64)
    x = layers.Conv3D(64, 3, strides=2, activation='relu', padding='same')(x)

    # (4, 4, 4, 64)-->(1, 1, 1, 64)
    x = layers.MaxPooling3D((int(x[0].shape[0]), int(x[0].shape[1]), int(x[0].shape[2])))(x)

    # 展平层：(1,1,1,64)-->(64)
    x = layers.Flatten()(x)

    # 全连接层+输出层
    x = layers.Dense(units=32, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=16, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=4, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units=1, activation='relu')(x)

    # 构建模型
    model = keras.Model(inputs, outputs)
    model.summary()
    return model
    # 可以考虑zeropadding3d层


def build_model_106(input_shape, window):
    # 输入层:(100,109,109,3)
    inputs = keras.Input(shape=input_shape)

    # CBS(k=3,s=1):(100,109,109,3)-->(50,55,55,16)
    x = layers.Conv3D(16, 3, strides=2, padding='same')(inputs)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # (50,55,55,16)-->(25,28,28,16)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (25,28,28,16)-->(13,14,14,32)
    x = layers.Conv3D(32, 3, strides=2, padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # (13,14,14,32)-->(7,7,7,32)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (7,7,7,32) -->(4, 4, 4, 64)
    x = layers.Conv3D(64, 3, strides=2, padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # (4, 4, 4, 64)-->(1, 1, 1, 64)
    x = layers.MaxPooling3D((int(x[0].shape[0]), int(x[0].shape[1]), int(x[0].shape[2])))(x)

    # 展平层：(1,1,1,64)-->(64)
    x = layers.Flatten()(x)

    # 全连接层+输出层
    x = layers.Dense(units=32, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=16, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=4, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units=1, activation='relu')(x)

    # 构建模型
    model = keras.Model(inputs, outputs)
    model.summary()
    return model
    # 可以考虑zeropadding3d层
    x = layers.Concatenate()(x)
    x = layers.merge()


def build_model_107(input_shape, input_shape1, input_shape2, window):
    # 输入层:(100,109,109,3)
    inputs = keras.Input(shape=input_shape)
    inputs_IR = keras.Input(shape=input_shape1)
    inputs_CT = keras.Input(shape=input_shape2)

    # CBS(k=3,s=1):(100,109,109,3)-->(50,55,55,16)
    x = layers.Conv3D(16, 3, strides=2, padding='same')(inputs)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # (50,55,55,16)-->(25,28,28,16)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (25,28,28,16)-->(13,14,14,32)
    x = layers.Conv3D(32, 3, strides=2, padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # (13,14,14,32)-->(7,7,7,32)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (7,7,7,32) -->(4, 4, 4, 64)
    x = layers.Conv3D(64, 3, strides=2, padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # (4, 4, 4, 64)-->(1, 1, 1, 64)
    x = layers.MaxPooling3D((int(x[0].shape[0]), int(x[0].shape[1]), int(x[0].shape[2])))(x)

    # 展平层：(1,1,1,64)-->(64)
    x = layers.Flatten()(x)
    x = layers.Dense(units=window, activation='relu')(x)
    x = layers.Reshape(target_shape=(x.shape[1], 1))(x)
    inputs_IR = layers.Reshape(target_shape=(inputs_IR.shape[1], 1))(inputs_IR)
    inputs_CT = layers.Reshape(target_shape=(inputs_CT.shape[1], 1))(inputs_CT)

    x = layers.Concatenate()([x, inputs_IR, inputs_CT])
    x = layers.Reshape(target_shape=(x.shape[1], x.shape[2], 1))(x)
    x = layers.Conv2D(3, strides=1, )(x)

    # 全连接层+输出层
    x = layers.Dense(units=32, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=16, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=4, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units=1, activation='relu')(x)

    # 构建模型
    model = keras.Model(inputs, outputs)
    model.summary()
    return model
    # 可以考虑zeropadding3d层
    # x = layers.Concatenate()(x)
    # x = layers.merge()


def build_model_108(input_shape, input_shape_a, input_shape_b, window):
    # 输入层:(100,109,109,3)
    inputs = keras.Input(shape=input_shape)
    inputs_IR = keras.Input(shape=input_shape_a)
    inputs_CT = keras.Input(shape=input_shape_b)

    a = build_model_aux(inputs, inputs_IR)
    b = build_model_aux(inputs, inputs_CT)

    # CBS(k=3,s=1):(100,109,109,3)-->(50,55,55,16)
    x = layers.Conv3D(16, 3, strides=2, padding='same')(inputs)
    x = layers.Activation('relu')(x)

    # (50,55,55,16)-->(25,28,28,16)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (25,28,28,16)-->(13,14,14,32)
    x = layers.Conv3D(32, 3, strides=2, padding='same')(x)
    x = layers.Activation('relu')(x)
    # (13,14,14,32)-->(7,7,7,32)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (7,7,7,32) -->(4, 4, 4, 64)
    x = layers.Conv3D(64, 3, strides=2, padding='same')(x)
    x = layers.Activation('relu')(x)

    # (4, 4, 4, 64)-->(1, 1, 1, 64)
    x = layers.MaxPooling3D((int(x[0].shape[0]), int(x[0].shape[1]), int(x[0].shape[2])))(x)

    # 展平层：(1,1,1,64)-->(64)
    x = layers.Flatten()(x)

    # 全连接层+输出层
    x = layers.Dense(units=32, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=16, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=8, activation='relu')(x)
    x = layers.Concatenate()([x, inputs_IR, inputs_CT])
    # x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=8, activation='relu')(x)
    x = layers.Dense(units=4, activation='relu')(x)
    outputs = layers.Dense(units=1, activation='relu')(x)

    # 构建模型
    model = keras.Model(inputs=[inputs, inputs_IR, inputs_CT], outputs=outputs)
    model.summary()
    return model
    # 可以考虑zeropadding3d层


def build_model_109(input_shape, input_shape_a, input_shape_b, input_shape_c, window):
    # 输入层:(100,109,109,3)
    inputs = keras.Input(shape=input_shape)
    inputs_a = keras.Input(shape=input_shape_a)
    inputs_b = keras.Input(shape=input_shape_b)
    inputs_c = keras.Input(shape=input_shape_c)

    a = build_model_aux(inputs, inputs_a)
    b = build_model_aux(inputs, inputs_b)
    c = build_model_aux(inputs, inputs_c)

    # CBS(k=3,s=1):(100,109,109,3)-->(50,55,55,16)
    x = layers.Conv3D(16, 3, strides=2, padding='same')(inputs)
    x = layers.Activation('relu')(x)

    # (50,55,55,16)-->(25,28,28,16)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (25,28,28,16)-->(13,14,14,32)
    x = layers.Conv3D(32, 3, strides=2, padding='same')(x)
    x = layers.Activation('relu')(x)
    # (13,14,14,32)-->(7,7,7,32)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (7,7,7,32) -->(4, 4, 4, 64)
    x = layers.Conv3D(64, 3, strides=2, padding='same')(x)
    x = layers.Activation('relu')(x)

    # (4, 4, 4, 64)-->(1, 1, 1, 64)
    x = layers.MaxPooling3D((int(x[0].shape[0]), int(x[0].shape[1]), int(x[0].shape[2])))(x)

    # 展平层：(1,1,1,64)-->(64)
    x = layers.Flatten()(x)

    # 全连接层+输出层
    x = layers.Dense(units=32, activation='relu')(x)
    x = layers.Dense(units=16, activation='relu')(x)
    x = layers.Dense(units=8, activation='relu')(x)
    x = layers.Dense(units=4, activation='relu')(x)
    x = layers.Dense(units=1, activation='relu')(x)
    x = layers.Concatenate()([x, a, b, c])
    outputs = layers.Dense(units=1, activation='relu')(x)

    # 构建模型
    model = keras.Model(inputs=[inputs, inputs_a, inputs_b, inputs_c], outputs=outputs)
    model.summary()
    return model
    # 可以考虑zeropadding3d层


def build_model_aux(inputs, inputs_aux):
    # 处理辅助变量模型
    # (50,)-->(50,1)
    t = layers.Reshape(target_shape=(inputs.shape[1], 1))(inputs_aux)
    # (50,1)-->(48,8)
    t = layers.Conv1D(8, 3, activation='relu')(t)
    # (48,8)-->(24,8)
    t = layers.MaxPool1D(2)(t)
    # (24, 8)-->(22,16)
    t = layers.Conv1D(16, 3, activation='relu')(t)
    # (22,16)-->(11,16)
    t = layers.MaxPool1D(2)(t)
    # (11,16)-->(9,32)
    t = layers.Conv1D(32, 3, activation='relu')(t)
    # (9,32)-->(5,32)
    t = layers.MaxPool1D(2)(t)
    # (9,32)-->(5,32)
    t = layers.Flatten()(t)
    # t = layers.Dense(units=64, activation='relu')(t)
    # t = layers.Dense(units=32, activation='relu')(t)
    # t = layers.Dense(units=16, activation='relu')(t)
    # t = layers.Dense(units=8, activation='relu')(t)
    # t = layers.Dense(units=4, activation='relu')(t)
    # t = layers.Dense(units=1, activation='relu')(t)
    return t


def build_model_110(input_shape, input_shape_a, input_shape_b, input_shape_c, window):
    # 输入层:(100,109,109,3)
    inputs = keras.Input(shape=input_shape)
    inputs_a = keras.Input(shape=input_shape_a)
    inputs_b = keras.Input(shape=input_shape_b)
    inputs_c = keras.Input(shape=input_shape_c)

    a = build_model_aux(inputs, inputs_a)
    b = build_model_aux(inputs, inputs_b)
    c = build_model_aux(inputs, inputs_c)

    # CBS(k=3,s=1):(100,109,109,3)-->(50,55,55,16)
    x = layers.Conv3D(16, 3, strides=2, padding='same')(inputs)
    x = layers.Activation('relu')(x)

    # (50,55,55,16)-->(25,28,28,16)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (25,28,28,16)-->(13,14,14,32)
    x = layers.Conv3D(32, 3, strides=2, padding='same')(x)
    x = layers.Activation('relu')(x)
    # (13,14,14,32)-->(7,7,7,32)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (7,7,7,32) -->(4, 4, 4, 64)
    x = layers.Conv3D(64, 3, strides=2, padding='same')(x)
    x = layers.Activation('relu')(x)

    # (4, 4, 4, 64)-->(1, 1, 1, 64)
    x = layers.MaxPooling3D((int(x[0].shape[0]), int(x[0].shape[1]), int(x[0].shape[2])))(x)

    # 展平层：(1,1,1,64)-->(64)
    x = layers.Flatten()(x)

    # 全连接层+输出层：(448)
    x = layers.Concatenate()([x, a, b, c])

    x = layers.Dense(units=256, activation='relu')(x)
    x = layers.Dense(units=64, activation='relu')(x)
    x = layers.Dense(units=16, activation='relu')(x)
    x = layers.Dense(units=4, activation='relu')(x)
    outputs = layers.Dense(units=1, activation='relu')(x)

    # 构建模型
    model = keras.Model(inputs=[inputs, inputs_a, inputs_b, inputs_c], outputs=outputs)
    model.summary()
    return model
    # 可以考虑zeropadding3d层


def build_model_aux_111(inputs, inputs_aux):
    # 处理辅助变量模型
    # (50,)-->(50,1)
    t = layers.Reshape(target_shape=(inputs.shape[1], 1))(inputs_aux)
    # (50,1)-->(50,16)
    t = layers.Conv1D(16, 1, padding='same', activation='relu')(t)
    # (50,16)-->(25,16)
    t = layers.MaxPool1D(2)(t)
    # (25,16)-->(25,16)
    t = layers.Conv1D(32, 3, padding='same', activation='relu')(t)
    # (25,32)-->(13,32)
    t = layers.MaxPool1D(2)(t)
    # (13,32)-->(13,64)
    t = layers.Conv1D(64, 3, padding='same', activation='relu')(t)
    # (13,64)-->(6,64)
    t = layers.MaxPool1D(2)(t)
    # (6,64)-->(6,128)
    t = layers.Conv1D(128, 3, padding='same', activation='relu')(t)
    # (6,128)-->(3,128)
    t = layers.MaxPool1D(2)(t)
    # (3,128)-->(384)
    t = layers.Flatten()(t)
    t = layers.Dense(units=128, activation='relu')(t)
    return t


def build_model_111(input_shape, input_shape_a, input_shape_b, input_shape_c, window):
    # 输入层:(50,109,109,3)
    inputs = keras.Input(shape=input_shape)
    inputs_a = keras.Input(shape=input_shape_a)
    inputs_b = keras.Input(shape=input_shape_b)
    inputs_c = keras.Input(shape=input_shape_c)

    a = build_model_aux_111(inputs, inputs_a)
    b = build_model_aux_111(inputs, inputs_b)
    c = build_model_aux_111(inputs, inputs_c)

    # CBS(k=3,s=1):(50,109,109,3)-->(50,109,109,32)
    x = layers.Conv3D(32, 3, padding='same', activation='relu')(inputs)

    # (50,109,109,32)-->(25,55,55,32)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (25,55,55,32)-->(25,55,55,32)
    x = layers.Conv3D(64, 3, padding='same', activation='relu')(x)

    # (25,55,55,64)-->(12,28,28,64)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (12,28,28,64)-->(12,28,28,128)
    x = layers.Conv3D(128, 3, padding='same', activation='relu')(x)

    # (12,28,28,128)-->(6,14,14,128)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (6,14,14,128)-->(6,14,14,256)
    x = layers.Conv3D(256, 3, padding='same', activation='relu')(x)

    # (6,14,14,256)-->(3,7,7,256)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # (3,7,7,256)-->(3,7,7,512)
    x = layers.Conv3D(512, 3, padding='same', activation='relu')(x)

    # (3,7,7,512)-->(1,3,3,256)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    # 展平层：(1,1,1,64)-->(64)
    x = layers.Flatten()(x)
    x = layers.Dense(units=1024, activation='relu')(x)
    x = layers.Dense(units=512, activation='relu')(x)
    x = layers.Dense(units=256, activation='relu')(x)
    x = layers.Dense(units=128, activation='relu')(x)

    # 全连接层+输出层：(448)
    x = layers.Concatenate()([x, a, b, c])

    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dense(units=64, activation='relu')(x)
    x = layers.Dense(units=32, activation='relu')(x)
    x = layers.Dense(units=16, activation='relu')(x)
    x = layers.Dense(units=8, activation='relu')(x)
    x = layers.Dense(units=4, activation='relu')(x)
    outputs = layers.Dense(units=1, activation='relu')(x)

    # 构建模型
    model = keras.Model(inputs=[inputs, inputs_a, inputs_b, inputs_c], outputs=outputs)
    model.summary()
    return model
    # 可以考虑zeropadding3d层


# CBS模型块 备份
def model_CBR(t, channel, k):
    t = layers.Conv3D(channel, k, padding='same')(t)
    t = layers.BatchNormalization()(t)
    t = layers.Activation('relu')(t)
    return t

# def model_CBR(t, channel, k):
#     t = layers.Conv3D(channel, k, padding='same', activation='relu')(t)
#     return t

def model_CBR_1D(t, channel, k):
    t = layers.Conv1D(channel, k, padding='same')(t)
    t = layers.BatchNormalization()(t)
    t = layers.Activation('relu')(t)
    return t

# def model_CBR_1D(t, channel, k):
#     t = layers.Conv1D(channel, k, padding='same', activation='relu')(t)
#     return t


def model_MP(t, channel, k):
    t1 = layers.MaxPooling3D((k, k, k), padding='same')(t)
    t1 = layers.Conv3D(channel / 2, 1, strides=1)(t1)

    t2 = layers.Conv3D(channel / 2, 1, strides=1)(t)
    t2 = layers.Conv3D(channel / 2, 3, strides=2, padding='same')(t2)

    t = layers.Concatenate()([t1, t2])
    return t

# def model_MP(t, channel, k):
#     t = layers.MaxPooling3D((k, k, k), padding='same')(t)
#     return t


def model_MP_1D(t, channel, k):
    t1 = layers.MaxPooling1D(2, padding='same')(t)
    t1 = layers.Conv1D(channel / 2, 1, strides=1)(t1)

    t2 = layers.Conv1D(channel / 2, 1, strides=1)(t)
    t2 = layers.Conv1D(channel / 2, 3, strides=2, padding='same')(t2)

    t = layers.Concatenate()([t1, t2])
    return t

# def model_MP_1D(t, channel, k):
#     t = layers.MaxPooling1D(2, padding='same')(t)
#     return t


def build_model_aux_200(inputs, inputs_aux):
    # 处理辅助变量模型
    # (50,)-->(50,1)
    t = layers.Reshape(target_shape=(inputs.shape[1], 1))(inputs_aux)

    # -->(50,16)
    t = model_CBR_1D(t, 16, 3)
    # -->(25,16)
    t = model_MP_1D(t, 16, 2)

    # -->(25,32)
    t = model_CBR_1D(t, 32, 3)
    # -->(13,32)
    t = model_MP_1D(t, 32, 2)

    # -->(13,64)
    t = model_CBR_1D(t, 64, 3)
    # -->(7,64)
    t = model_MP_1D(t, 64, 2)

    # -->(7,128)
    t = model_CBR_1D(t, 128, 3)
    # -->(4,128)
    t = model_MP_1D(t, 128, 2)

    t = layers.Flatten()(t)
    t = layers.Dense(units=128, activation='relu')(t)
    return t


def build_model_200(input_shape, input_shape_a, input_shape_b, input_shape_c, window):
    # 输入层:(50,109,109,3)
    inputs = keras.Input(shape=input_shape)
    inputs_a = keras.Input(shape=input_shape_a)
    inputs_b = keras.Input(shape=input_shape_b)
    inputs_c = keras.Input(shape=input_shape_c)

    a = build_model_aux_200(inputs, inputs_a)
    b = build_model_aux_200(inputs, inputs_b)
    c = build_model_aux_200(inputs, inputs_c)

    # c1
    # CBR(k=3,s=1):(50,109,109,3)-->(50,109,109,16)
    x = layers.Conv3D(8, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # -->(50,55,55,16)
    x = model_MP(x, 8, 2)
    # c2
    # -->(50,55,55,32)
    x = model_CBR(x, 16, 3)
    # -->(25,28,28,32)
    x = model_MP(x, 16, 2)
    # c3
    # -->(25,28,28,64)
    x = model_CBR(x, 32, 3)
    # -->(13,14,14,64)
    x = model_MP(x, 32, 2)
    # c4
    # -->(13,14,14,128)
    x = model_CBR(x, 64, 3)
    # -->(7,7,7,128)
    x = model_MP(x, 64, 2)

    # c5
    # -->(7,7,7,256)
    x = model_CBR(x, 128, 3)
    # -->(4,4,4,256)
    x = model_MP(x, 128, 2)

    # 展平层：(1,1,1,64)-->(64)
    x = layers.Flatten()(x)
    x = layers.Dense(units=1024, activation='relu')(x)
    # x = layers.Dense(units=1024, activation='relu')(x)
    # x = layers.Dense(units=512, activation='relu')(x)
    x = layers.Dense(units=256, activation='relu')(x)
    # x = layers.Dense(units=128, activation='relu')(x)

    # 全连接层+输出层：(448)
    # x = layers.Concatenate()([x, a, b, c])
    x = layers.Concatenate()([x, a, b])

    # x = layers.Dense(units=1024, activation='relu')(x)
    x = layers.Dense(units=128, activation='relu')(x)
    # x = layers.Dense(units=64, activation='relu')(x)
    x = layers.Dense(units=32, activation='relu')(x)
    # x = layers.Dense(units=16, activation='relu')(x)
    x = layers.Dense(units=8, activation='relu')(x)
    # x = layers.Dense(units=4, activation='relu')(x)
    outputs = layers.Dense(units=1, activation='relu')(x)

    # 构建模型
    model = keras.Model(inputs=[inputs, inputs_a, inputs_b, inputs_c], outputs=outputs)
    model.summary()
    return model

def CNN_LSTM(input_shape, input_shape_a, input_shape_b, input_shape_c, window):
    # 输入层:(50,109,109,3)
    inputs = keras.Input(shape=input_shape)
    inputs_a = keras.Input(shape=input_shape_a)
    inputs_b = keras.Input(shape=input_shape_b)
    inputs_c = keras.Input(shape=input_shape_c)

    print(1)


