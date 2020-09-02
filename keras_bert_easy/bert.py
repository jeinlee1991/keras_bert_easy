import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.math_ops import erf, sqrt

from .layers import *
from .backend import keras
from .optimizers import AdamWarmup

here = os.path.abspath(os.path.dirname(__file__))

__all__ = [
    'TOKEN_PAD', 'TOKEN_UNK', 'TOKEN_CLS', 'TOKEN_SEP', 'TOKEN_MASK',
    'gelu', 'get_model', 'compile_model', 'get_base_dict',
    'get_custom_objects', 'set_custom_objects', 'build_pretrained_model',
]


TOKEN_PAD = ''  # Token for padding
TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_MASK = '[MASK]'  # Token for masking


def get_base_dict():
    """Get basic dictionary containing special tokens."""
    return {
        TOKEN_PAD: 0,
        TOKEN_UNK: 1,
        TOKEN_CLS: 2,
        TOKEN_SEP: 3,
        TOKEN_MASK: 4,
    }


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


def get_encoders(
        input_layer, encoder_num, head_num,
        intermediate_size, feed_forward_activation=gelu,
        dropout_rate=0.0,):
    """Get encoders.

    :param encoder_num: Number of encoder components.
    :param input_layer: Input layer.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """

    x0 = input_layer
    for i in range(encoder_num):
        name = 'Encoder-%d' % (i + 1)
        attention_name = '%s-MultiHeadSelfAttention' % name
        feed_forward_name = '%s-FeedForward' % name

        # attention
        x = MultiHeadAttention(
            head_num=head_num,
            name=attention_name,
            attention_prob_dropout_rate=dropout_rate
        )(x0)
        x = keras.layers.Dropout(
            rate=dropout_rate,
            name='%s-Dropout' % attention_name
        )(x)
        x = keras.layers.Add(name='%s-Add' % attention_name)([x0, x])
        x = LayerNormalization(name='%s-Norm' % attention_name)(x)

        # feedforward
        x0 = x
        x = FeedForward(
            units=intermediate_size,
            activation=feed_forward_activation,
            name=feed_forward_name
        )(x0)
        x = keras.layers.Dropout(
            rate=dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )(x)
        x = keras.layers.Add(name='%s-Add' % feed_forward_name)([x0, x])
        x = LayerNormalization(name='%s-Norm' % feed_forward_name)(x)

        x0 = x

    return x0


def get_model(
        token_num, pos_num=512, seq_len=512, embed_dim=768,
        transformer_num=12, head_num=12, intermediate_size=3072,
        dropout_rate=0.1, feed_forward_activation='gelu',
        trainable=True, mode='finetune',**kwargs):
    """Get BERT model.

    See: https://arxiv.org/pdf/1810.04805.pdf

    :param token_num: Number of tokens.
    :param pos_num: Maximum position.
    :param seq_len: Maximum length of the input sequence or None.
    :param embed_dim: Dimensions of embeddings.
    :param transformer_num: Number of transformers.
    :param head_num: Number of heads in multi-head attention in each transformer.
    :param intermediate_size: Dimension of the feed forward layer in each transformer.
    :param dropout_rate: Dropout rate.
    :param feed_forward_activation: Activation for feed-forward layers.
    :param mode: finetine or pretrain. A built model with MLM and NSP outputs will be returned if it is `pretrain`,
                otherwise model with transformer output will be returned.
    :param trainable: Whether the model is trainable.
    :return: The built model.
    """

    if feed_forward_activation == 'gelu':
        feed_forward_activation = gelu

    inputs = [keras.layers.Input(shape=(seq_len,), name='Input-%s' % name)
              for name in ['Token', 'Segment', 'Masked']]

    x, embed_weights = get_embedding(
        inputs,
        token_num=token_num,
        embed_dim=embed_dim,
        pos_num=pos_num,
    )

    x = keras.layers.Dropout(
        rate=dropout_rate,
        name='Embedding-Dropout'
    )(x)

    x = LayerNormalization(
        trainable=trainable,
        name='Embedding-Norm'
    )(x)

    x = get_encoders(
        input_layer=x,
        encoder_num=transformer_num,
        head_num=head_num,
        intermediate_size=intermediate_size,
        feed_forward_activation=feed_forward_activation,
        dropout_rate=dropout_rate,
    )

    if mode=='pretrain':
        mlm_dense_layer = keras.layers.Dense(
            units=embed_dim,
            activation=feed_forward_activation,
            name='MLM-Dense',
        )(x)
        mlm_norm_layer = LayerNormalization(name='MLM-Norm')(mlm_dense_layer)
        mlm_pred_layer = EmbeddingSimilarity(name='MLM-Sim')([mlm_norm_layer, embed_weights])
        masked_layer = Masked(name='MLM')([mlm_pred_layer, inputs[-1]])
        nsp_dense_layer = keras.layers.Dense(units=embed_dim, activation='tanh', name='NSP-Dense')(x[:,0,:])
        nsp_pred_layer = keras.layers.Dense(units=2, activation='softmax', name='NSP')(nsp_dense_layer)
        model = keras.models.Model(inputs=inputs, outputs=[masked_layer, nsp_pred_layer])
        for layer in model.layers:
            layer.trainable = trainable
        return model

    else:
        inputs = inputs[:2]
        model = keras.models.Model(inputs=inputs, outputs=x)
        for layer in model.layers:
            layer.trainable = trainable
        return model


def compile_model(
        model, weight_decay=0.01, decay_steps=100000,
        warmup_steps=10000, learning_rate=1e-4):
    """Compile the model with warmup optimizer and sparse cross-entropy loss.

    :param model: The built model.
    :param weight_decay: Weight decay rate.
    :param decay_steps: Learning rate will decay linearly to zero in decay steps.
    :param warmup_steps: Learning rate will increase linearly to learning_rate in first warmup steps.
    :param learning_rate: Learning rate.
    :return: The compiled model.
    """
    model.compile(
        optimizer=AdamWarmup(
            decay_steps=decay_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_decay_pattern=['embeddings', 'kernel', 'W1', 'W2', 'Wk', 'Wq', 'Wv', 'Wo'],
        ),
        loss=keras.losses.sparse_categorical_crossentropy,
    )


def get_default_config_file(modelname='bert'):
    file = here + '/configs/bert/bert_config.json'
    if modelname=='electra-small':
        file = here + '/configs/electra-small/electra_small_config.json'
    return file


def load_from_ckpt(model, config, checkpoint_file, mode='finetune'):
    """Load trained official model from checkpoint.

    :param model: Built keras model.
    :param config: Loaded configuration file.
    :param checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
    :param mode: `finetune` or `pretrain`
    """
    ckpt_reader = tf.train.load_checkpoint(checkpoint_file)
    def loader(name):
        if name.endswith(":0"):
            name = name[:-2]
        return ckpt_reader.get_tensor(name)

    # map from keras-layer to ckpt's variable
    variable_map = {
        'Embedding-Token': ['bert/embeddings/word_embeddings'],
        'Embedding-Position': ['bert/embeddings/position_embeddings'],
        'Embedding-Segment': ['bert/embeddings/token_type_embeddings'],
        'Embedding-Norm': ['bert/embeddings/LayerNorm/gamma', 'bert/embeddings/LayerNorm/beta'],
    }

    for i in range(config['num_hidden_layers']):
        layername = 'Encoder-%d-MultiHeadSelfAttention'%(i + 1)
        variable_map[layername] = [
            'bert/encoder/layer_%d/attention/self/query/kernel' % i,
            'bert/encoder/layer_%d/attention/self/query/bias' % i,
            'bert/encoder/layer_%d/attention/self/key/kernel' % i,
            'bert/encoder/layer_%d/attention/self/key/bias' % i,
            'bert/encoder/layer_%d/attention/self/value/kernel' % i,
            'bert/encoder/layer_%d/attention/self/value/bias' % i,
            'bert/encoder/layer_%d/attention/output/dense/kernel' % i,
            'bert/encoder/layer_%d/attention/output/dense/bias' % i,
        ]
        layername = 'Encoder-%d-MultiHeadSelfAttention-Norm'%(i + 1)
        variable_map[layername] = [
            'bert/encoder/layer_%d/attention/output/LayerNorm/gamma' % i,
            'bert/encoder/layer_%d/attention/output/LayerNorm/beta' % i,
        ]
        layername = 'Encoder-%d-FeedForward' % (i + 1)
        variable_map[layername] = [
            'bert/encoder/layer_%d/intermediate/dense/kernel' % i,
            'bert/encoder/layer_%d/intermediate/dense/bias' % i,
            'bert/encoder/layer_%d/output/dense/kernel' % i,
            'bert/encoder/layer_%d/output/dense/bias' % i,
        ]
        layername = 'Encoder-%d-FeedForward-Norm' % (i + 1)
        variable_map[layername] = [
            'bert/encoder/layer_%d/output/LayerNorm/gamma' % i,
            'bert/encoder/layer_%d/output/LayerNorm/beta' % i,
        ]

    if mode=='pretrain':
        variable_map['MLM-Dense'] = [
            'cls/predictions/transform/dense/kernel',
            'cls/predictions/transform/dense/bias',
        ]
        variable_map['MLM-Norm'] = [
            'cls/predictions/transform/LayerNorm/gamma',
            'cls/predictions/transform/LayerNorm/beta',
        ]
        variable_map['MLM-Sim'] = ['cls/predictions/output_bias']
        variable_map['NSP-Dense'] = ['bert/pooler/dense/kernel', 'bert/pooler/dense/bias']
        variable_map['NSP'] = [
            'cls/seq_relationship/output_weights',
            'cls/seq_relationship/output_bias'
        ]

    valid_layernames = [layer.name for layer in model.layers]
    weights_value_pairs = []
    for layername, v in variable_map.items():
        if layername not in valid_layernames:
            continue
        weights = model.get_layer(layername).trainable_weights
        values = [loader(item) for item in v]
        if layername == 'NSP':
            values[0] = np.transpose(values[0])
        weights_value_pairs += list(zip(weights, values))

    K.batch_set_value(weights_value_pairs)


def build_pretrained_model(
        config_file=None, checkpoint_file=None, trainable=True,
        seq_len=None, mode='finetune', **kwargs):
    """
    :param config_file: The path to the JSON configuration file.
    :param checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
    :param trainable: Whether the model is trainable. The default value is the same with `training`.
    :param seq_len: If it is not None and it is shorter than the value in the config file, the weights in
                    position embeddings will be sliced to fit the new length.
    :param mode: `finetune` or `pretrain`
    :return: model
    """
    if not config_file:
        config_file = get_default_config_file(modelname='bert')

    with open(config_file, 'r') as reader:
        config = json.loads(reader.read())

    if seq_len is not None:
        if seq_len > config['max_position_embeddings']:
            raise ValueError('seq_len must not be greater than %s, got %s'%
                             (config['max_position_embeddings'], seq_len))

    model = get_model(
        token_num=config['vocab_size'],
        pos_num=config['max_position_embeddings'],
        seq_len=seq_len,
        embed_dim=config['hidden_size'],
        transformer_num=config['num_hidden_layers'],
        head_num=config['num_attention_heads'],
        intermediate_size=config['intermediate_size'],
        feed_forward_activation=config['hidden_act'],
        trainable=trainable,
        mode=mode,
        **kwargs)

    if checkpoint_file:
        load_from_ckpt(model, config, checkpoint_file, mode=mode)
    return model


def get_custom_objects():
    """Get all custom objects for loading saved models."""
    custom_objects = {}
    custom_objects['PositionEmbedding'] = PositionEmbedding
    custom_objects['TokenEmbedding'] = TokenEmbedding
    custom_objects['EmbeddingSimilarity'] = EmbeddingSimilarity
    custom_objects['Masked'] = Masked
    custom_objects['gelu'] = gelu
    custom_objects['AdamWarmup'] = AdamWarmup
    return custom_objects


def set_custom_objects():
    """Add custom objects to Keras environments."""
    for k, v in get_custom_objects().items():
        keras.utils.get_custom_objects()[k] = v

