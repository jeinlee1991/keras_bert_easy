import os
os.environ['TF_KERAS'] = '0'
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

try:
    from . import layerwiseLR
    from .layerwiseLR import keras
except:
    import layerwiseLR
    from layerwiseLR import keras


def test_keras_Adam_2lr(use_accum_opt=False):
    print('\n--------------------\nbegin to test keras_Adam_2lr...')

    # tf 2.x 需要禁掉Eager
    if tf.__version__.startswith('2.'):
        tf.compat.v1.disable_eager_execution()

    # get model
    inpl = keras.layers.Input(shape=[5,], name='early-layer1')
    x = keras.layers.Embedding(1000, 10, name='early-layer2')(inpl)
    x = keras.layers.Dense(10, name='early-layer3')(x)
    x = keras.layers.Dense(10, name='final-layer1')(x)
    x = keras.layers.Lambda(lambda x:x[:,0,:])(x)
    output = keras.layers.Dense(1, name='final-layer2')(x)
    model = keras.Model(inpl, output)
    # model.summary()

    # get final_layers_weights
    final_layers_names = ['final-layer1', 'final-layer2']
    final_layers_weights= []
    for name in final_layers_names:
        final_layers_weights += model.get_layer(name).weights
    print('final_layers_weights: ', [item.name for item in final_layers_weights])

    # train
    optimizer = layerwiseLR.keras_Adam_2lr(
        final_layers=final_layers_weights,
        lr=(1e-3, 1e-2),  # early_layers, final_layers 学习率分别设为 1e-3, 1e-2
        weight_decay=0.01,
    )
    if use_accum_opt:
        optimizer = layerwiseLR.AccumOptimizer(optimizer=optimizer, steps_per_update=16)
        print('using AccumOptimizer...')

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.binary_crossentropy
    )

    x_train = np.random.randint(0, 1000, size=[100,5])  # data size: 100
    y_train = np.array([1, 0]*50)
    model.fit(x_train, y_train, epochs=3, verbose=2)

    # test success!
    # print('tf version: ', tf.__version__)
    # print('keras version: ', keras.__version__)
    print('test keras_Adam_2lr ok!')


def test_keras_Adam_3lr(use_accum_opt=False):
    print('\n--------------------\nbegin to test keras_Adam_3lr...')

    # tf 2.x 需要禁掉Eager
    if tf.__version__.startswith('2.'):
        tf.compat.v1.disable_eager_execution()

    # get model
    inpl = keras.layers.Input(shape=[5,], name='early-layer1')
    x = keras.layers.Dense(10, name='early-layer2')(inpl)
    x = keras.layers.Dense(10, name='middle-layer1')(x)
    x = keras.layers.Dense(10, name='middle-layer2')(x)
    x = keras.layers.Dense(10, name='final-layer1')(x)
    output = keras.layers.Dense(1, name='final-layer2')(x)
    model = keras.Model(inpl, output)
    # model.summary()

    # get middle_layers_weights, final_layers_weights
    middle_layers_names = ['middle-layer1', 'middle-layer2']
    final_layers_names = ['final-layer1', 'final-layer2']
    middle_layers_weights, final_layers_weights= [], []
    for name in middle_layers_names:
        middle_layers_weights += model.get_layer(name).weights
    for name in final_layers_names:
        final_layers_weights += model.get_layer(name).weights
    print('middle_layers_weights: ', [item.name for item in middle_layers_weights])
    print('final_layers_weights: ', [item.name for item in final_layers_weights])

    # train
    optimizer = layerwiseLR.keras_Adam_3lr(
        middle_layers=middle_layers_weights,
        final_layers=final_layers_weights,
        lr=(1e-3, 1e-2, 1e-1),  # early_layers, middle_layers, final_layers 学习率分别设为 1e-3, 1e-2, 1e-1
        weight_decay=0.01,
    )
    if use_accum_opt:
        optimizer = layerwiseLR.AccumOptimizer(optimizer=optimizer, steps_per_update=16)
        print('using AccumOptimizer...')
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.binary_crossentropy
    )

    x_train = np.random.random(size=[100,5])  # data size: 100
    y_train = np.array([1, 0]*50)
    model.fit(x_train, y_train, epochs=3, verbose=0)

    # test success!
    print('test keras_Adam_3lr ok!')


def test_keras_Adam_lr_decay(use_accum_opt=False):
    print('\n--------------------\nbegin to test keras_Adam_lr_decay...')

    # tf 2.x 需要禁掉Eager
    if tf.__version__.startswith('2.'):
        tf.compat.v1.disable_eager_execution()

    # get model
    inpl = keras.layers.Input(shape=[5,], name='layer1')
    x = keras.layers.Dense(10, name='layer2')(inpl)
    x = keras.layers.Dense(10, name='layer3')(x)
    x = keras.layers.Dense(10, name='layer4')(x)
    x = keras.layers.Dense(10, name='layer5')(x)
    output = keras.layers.Dense(1, name='layer6')(x)
    model = keras.Model(inpl, output)
    # model.summary()

    n_layers = len(model.layers)
    weight_name_to_layer_depth = layerwiseLR.keras_Adam_lr_decay.build_dict_layer_depth(model)

    # train
    optimizer = layerwiseLR.keras_Adam_lr_decay(
        weight_name_to_layer_depth=weight_name_to_layer_depth,
        n_layers=n_layers,
        layerwise_lr_decay_power=0.8,  # 学习率逐层衰减的权重
        lr=1e-3,  # 基础学习率，即模型最后一层的学习率，其他层的学习率在此基础上衰减
        weight_decay=0.01,
    )
    if use_accum_opt:
        optimizer = layerwiseLR.AccumOptimizer(optimizer=optimizer, steps_per_update=16)
        print('using AccumOptimizer...')
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.binary_crossentropy
    )

    x_train = np.random.random(size=[100,5])  # data size: 100
    y_train = np.array([1, 0]*50)
    model.fit(x_train, y_train, epochs=3, verbose=0)

    # test success!
    print('test keras_Adam_lr_decay ok!')


if __name__=='__main__':
    test_keras_Adam_2lr(use_accum_opt=True)
    # test_keras_Adam_3lr(use_accum_opt=True)
    # test_keras_Adam_lr_decay(use_accum_opt=True)

