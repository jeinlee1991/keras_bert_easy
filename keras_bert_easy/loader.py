import os
import json
import codecs
import time

import numpy as np
import tensorflow as tf
from .backend import keras, K
from .bert import get_model

__all__ = [
    'load_from_ckpt',
    'build_pretrained_model',
    'load_vocabulary',
]

here = os.path.abspath(os.path.dirname(__file__))


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

    weights_value_pairs = []
    for layername, v in variable_map.items():
        weights = model.get_layer(layername).trainable_weights
        values = [loader(item) for item in v]
        if layername == 'NSP':
            values[0] = np.transpose(values[0])
        weights_value_pairs += list(zip(weights, values))

    K.batch_set_value(weights_value_pairs)


def build_pretrained_model(config_file=None,
                           checkpoint_file=None,
                           trainable=True,
                           seq_len=None,
                           mode='finetune',
                           **kwargs):
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
        feed_forward_dim=config['intermediate_size'],
        feed_forward_activation=config['hidden_act'],
        mode=mode,
        trainable=trainable,
        **kwargs)

    if checkpoint_file:
        load_from_ckpt(model, config, checkpoint_file, mode=mode)
    return model


def load_vocabulary(vocab_path):
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict