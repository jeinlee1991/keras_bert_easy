import numpy as np
import tensorflow as tf
import keras


try:
    from . import layerwiseLR
except:
    import layerwiseLR


def test_keras_Adam_2lr():
    print('\n--------------------\nbegin to test keras_Adam_2lr...')

    # tf 2.x 需要禁掉Eager
    if tf.__version__.startswith('2.'):
        tf.compat.v1.disable_eager_execution()

    # get model
    inpl = keras.layers.Input(shape=[5,], name='early-layer1')
    x = keras.layers.Dense(10, name='early-layer2')(inpl)
    x = keras.layers.Dense(10, name='final-layer1')(x)
    output = keras.layers.Dense(1, name='final-layer2')(x)
    model = keras.Model(inpl, output)
    model.summary()

    # get final_layers_weights
    final_layers_names = ['final-layer1', 'final-layer2']
    final_layers_weights= []
    for name in final_layers_names:
        final_layers_weights += model.get_layer(name).weights
    print('final_layers_weights: ', [item.name for item in final_layers_weights])

    # train
    optimizer = layerwiseLR.keras_Adam_2lr(
        final_layers=final_layers_weights,
        lr=(1e-3, 1e-2)  # early_layers, final_layers 学习率分别设为 1e-3, 1e-2
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.binary_crossentropy
    )
    print('model compile done!')
    x_train = np.random.random(size=[100,5])  # data size: 100
    y_train = np.array([1, 0]*50)
    model.fit(x_train, y_train, epochs=3, verbose=2)

    # test success!
    print('tf version: ', tf.__version__)
    print('keras version: ', keras.__version__)
    print('test_keras_Adam_2lr ok!')


def test_tfkeras_Adam_2lr():
    print('\n--------------------\nbegin to test tfkeras_Adam_2lr...')

    # tf 2.x 需要禁掉Eager
    if tf.__version__.startswith('2.'):
        tf.compat.v1.disable_eager_execution()

    # get model
    inpl = keras.layers.Input(shape=[5,], name='early-layer1')
    x = keras.layers.Dense(10, name='early-layer2')(inpl)
    x = keras.layers.Dense(10, name='final-layer1')(x)
    output = keras.layers.Dense(1, name='final-layer2')(x)
    model = keras.Model(inpl, output)
    model.summary()

    # get final_layers_weights
    final_layers_names = ['final-layer1', 'final-layer2']
    final_layers_weights= []
    for name in final_layers_names:
        final_layers_weights += model.get_layer(name).weights
    print('final_layers_weights: ', [item.name for item in final_layers_weights])

    # train
    optimizer = layerwiseLR.keras_Adam_2lr(
        final_layers=final_layers_weights,
        lr=(1e-3, 1e-2)  # early_layers, final_layers 学习率分别设为 1e-3, 1e-2
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.binary_crossentropy
    )
    print('model compile done!')
    x_train = np.random.random(size=[100,5])  # data size: 100
    y_train = np.array([1, 0]*50)
    model.fit(x_train, y_train, epochs=3, verbose=2)

    # test success!
    print('tf version: ', tf.__version__)
    print('keras version: ', keras.__version__)
    print('test_tfkeras_Adam_2lr ok!')


def test_keras_Adam_3lr():
    print('\n--------------------\nbegin to test keras_Adam_3lr...')

    # tf 2.x 需要禁掉Eager
    if tf.__version__.startswith('2.'):
        tf.compat.v1.disable_eager_execution()
        print('disable eager done!')

    # get model
    inpl = keras.layers.Input(shape=[5,], name='early-layer1')
    x = keras.layers.Dense(10, name='early-layer2')(inpl)
    x = keras.layers.Dense(10, name='middle-layer1')(x)
    x = keras.layers.Dense(10, name='middle-layer2')(x)
    x = keras.layers.Dense(10, name='final-layer1')(x)
    output = keras.layers.Dense(1, name='final-layer2')(x)
    model = keras.Model(inpl, output)
    model.summary()

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
        lr=(1e-3, 1e-2, 1e-1)  # early_layers, middle_layers, final_layers 学习率分别设为 1e-3, 1e-2, 1e-1
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.binary_crossentropy
    )
    print('model compile done!')
    x_train = np.random.random(size=[100,5])  # data size: 100
    y_train = np.array([1, 0]*50)
    model.fit(x_train, y_train, epochs=3, verbose=2)

    # test success!
    print('tf version: ', tf.__version__, ', keras version: ', keras.__version__)
    print('test_keras_Adam_3lr ok!')


def test_tfkeras_Adam_3lr():
    print('\n--------------------\nbegin to test tfkeras_Adam_3lr...')

    # tf 2.x 需要禁掉Eager
    if tf.__version__.startswith('2.'):
        tf.compat.v1.disable_eager_execution()

    # get model
    inpl = tf.keras.layers.Input(shape=[5,], name='early-layer1')
    x = tf.keras.layers.Dense(10, name='early-layer2')(inpl)
    x = tf.keras.layers.Dense(10, name='middle-layer1')(x)
    x = tf.keras.layers.Dense(10, name='middle-layer2')(x)
    x = tf.keras.layers.Dense(10, name='final-layer1')(x)
    output = tf.keras.layers.Dense(1, name='final-layer2')(x)
    model = tf.keras.Model(inpl, output)
    model.summary()

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
    optimizer = layerwiseLR.tfkeras_Adam_3lr(
        middle_layers=middle_layers_weights,
        final_layers=final_layers_weights,
        lr=(1e-3, 1e-2, 1e-1)  # early_layers, middle_layers, final_layers 学习率分别设为 1e-3, 1e-2, 1e-1
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.binary_crossentropy
    )
    print('model compile done!')
    x_train = np.random.random(size=[100,5])  # data size: 100
    y_train = np.array([1, 0]*50)
    model.fit(x_train, y_train, epochs=3, verbose=2)

    # test success!
    print('tf version: ', tf.__version__, ', keras version: ', keras.__version__)
    print('test_tfkeras_Adam_3lr ok!')


if __name__=='__main__':
    test_keras_Adam_2lr()
    test_tfkeras_Adam_2lr()
    test_keras_Adam_3lr()
    test_tfkeras_Adam_3lr()
