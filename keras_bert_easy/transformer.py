import math
import numpy as np
import tensorflow as tf

from .layers import LayerNormalization
from .backend import keras, TF_KERAS, K


__all__ = [
    'get_custom_objects', 'get_encoders', 'get_decoders', 'get_model', 'decode',
    'attention_builder', 'feed_forward_builder', 'get_encoder_component', 'get_decoder_component',
]


def gelu(x):
    """An approximation of gelu.

    See: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


class TrigPosEmbedding(keras.layers.Layer):
    """Position embedding use sine and cosine functions.

    See: https://arxiv.org/pdf/1706.03762

    Expand mode:
        # Input shape
            2D tensor with shape: `(batch_size, sequence_length)`.

        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

    Add mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

    Concat mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim + output_dim)`.
    """
    MODE_EXPAND = 'expand'
    MODE_ADD = 'add'
    MODE_CONCAT = 'concat'

    def __init__(self,
                 mode=MODE_ADD,
                 output_dim=None,
                 **kwargs):
        """
        :param output_dim: The embedding dimension.
        :param kwargs:
        """
        if mode in [self.MODE_EXPAND, self.MODE_CONCAT]:
            if output_dim is None:
                raise NotImplementedError('`output_dim` is required in `%s` mode' % mode)
            if output_dim % 2 != 0:
                raise NotImplementedError('It does not make sense to use an odd output dimension: %d' % output_dim)
        self.mode = mode
        self.output_dim = output_dim
        self.supports_masking = True
        super(TrigPosEmbedding, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'mode': self.mode,
            'output_dim': self.output_dim,
        }
        base_config = super(TrigPosEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        if self.mode == self.MODE_EXPAND:
            return input_shape + (self.output_dim,)
        if self.mode == self.MODE_CONCAT:
            return input_shape[:-1] + (input_shape[-1] + self.output_dim,)
        return input_shape

    def call(self, inputs, mask=None):
        input_shape = K.shape(inputs)
        if self.mode == self.MODE_ADD:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], input_shape[2]
            pos_input = K.tile(K.expand_dims(K.arange(0, seq_len), axis=0), [batch_size, 1])
        elif self.mode == self.MODE_CONCAT:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], self.output_dim
            pos_input = K.tile(K.expand_dims(K.arange(0, seq_len), axis=0), [batch_size, 1])
        else:
            output_dim = self.output_dim
            pos_input = inputs
        if K.dtype(pos_input) != K.floatx():
            pos_input = K.cast(pos_input, K.floatx())
        evens = K.arange(0, output_dim // 2) * 2
        odds = K.arange(0, output_dim // 2) * 2 + 1
        even_embd = K.sin(
            K.dot(
                K.expand_dims(pos_input, -1),
                K.expand_dims(1.0 / K.pow(
                    10000.0,
                    K.cast(evens, K.floatx()) / K.cast(output_dim, K.floatx())
                ), 0)
            )
        )
        odd_embd = K.cos(
            K.dot(
                K.expand_dims(pos_input, -1),
                K.expand_dims(1.0 / K.pow(
                    10000.0, K.cast((odds - 1), K.floatx()) / K.cast(output_dim, K.floatx())
                ), 0)
            )
        )
        embd = K.stack([even_embd, odd_embd], axis=-1)
        output = K.reshape(embd, [-1, K.shape(inputs)[1], output_dim])
        if self.mode == self.MODE_CONCAT:
            output = K.concatenate([inputs, output], axis=-1)
        if self.mode == self.MODE_ADD:
            output += inputs
        return output


class EmbeddingRet(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return [
            super(EmbeddingRet, self).compute_output_shape(input_shape),
            (self.input_dim, self.output_dim),
        ]

    def compute_mask(self, inputs, mask=None):
        return [
            super(EmbeddingRet, self).compute_mask(inputs, mask),
            None,
        ]

    def call(self, inputs):
        return [
            super(EmbeddingRet, self).call(inputs),
            self.embeddings + 0.0,
        ]


class EmbeddingSim(keras.layers.Layer):
    """Calculate similarity between features and token embeddings with bias term."""

    def __init__(self,
                 use_bias=True,
                 initializer='zeros',
                 regularizer=None,
                 constraint=None,
                 stop_gradient=False,
                 **kwargs):
        """Initialize the layer.

        :param output_dim: Same as embedding output dimension.
        :param use_bias: Whether to use bias term.
        :param initializer: Initializer for bias.
        :param regularizer: Regularizer for bias.
        :param constraint: Constraint for bias.
        :param stop_gradient: Whether to stop gradient for input embedding.
        :param kwargs: Arguments for parent class.
        """
        super(EmbeddingSim, self).__init__(**kwargs)
        self.supports_masking = True
        self.use_bias = use_bias
        self.initializer = keras.initializers.get(initializer)
        self.regularizer = keras.regularizers.get(regularizer)
        self.constraint = keras.constraints.get(constraint)
        self.stop_gradient = stop_gradient
        self.bias = None

    def get_config(self):
        config = {
            'use_bias': self.use_bias,
            'initializer': keras.initializers.serialize(self.initializer),
            'regularizer': keras.regularizers.serialize(self.regularizer),
            'constraint': keras.constraints.serialize(self.constraint),
            'stop_gradient': self.stop_gradient,
        }
        base_config = super(EmbeddingSim, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.use_bias:
            embed_shape = input_shape[1]
            token_num = int(embed_shape[0])
            self.bias = self.add_weight(
                shape=(token_num,),
                initializer=self.initializer,
                regularizer=self.regularizer,
                constraint=self.constraint,
                name='bias',
            )
        super(EmbeddingSim, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        feature_shape, embed_shape = input_shape
        token_num = embed_shape[0]
        return feature_shape[:-1] + (token_num,)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask[0]

    def call(self, inputs, mask=None, **kwargs):
        inputs, embeddings = inputs
        if self.stop_gradient:
            embeddings = K.stop_gradient(embeddings)
        outputs = K.dot(inputs, K.transpose(embeddings))
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias)
        return keras.activations.softmax(outputs)


class MultiHeadAttention(keras.layers.Layer):
    """多头注意力机制
    """
    def __init__(
        self,
        head_num,
        use_bias=True,
        attention_scale=True,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.head_num = head_num
        self.use_bias = use_bias
        self.attention_scale = attention_scale
        self.kernel_init = keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        h = input_shape[-1]
        self.dense_q = keras.layers.Dense(
                  units=h,
                  use_bias=self.use_bias,
                  kernel_initializer=self.kernel_init
        )
        self.dense_k = keras.layers.Dense(
                  units=h,
                  use_bias=self.use_bias,
                  kernel_initializer=self.kernel_init
        )
        self.dense_v = keras.layers.Dense(
                  units=h,
                  use_bias=self.use_bias,
                  kernel_initializer=self.kernel_init
        )
        self.dense_o = keras.layers.Dense(
                  units=h,
                  use_bias=self.use_bias,
                  kernel_initializer=self.kernel_init
        )

    def call(self, inputs, mask=None):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
        if isinstance(mask, list):
            mask = mask[0]

        shape = K.shape(q)
        key_size = shape[-1] // self.head_num

        # dense
        q = self.dense_q(q)
        k = self.dense_k(k)
        v = self.dense_v(v)

        # reshape according to head_num
        q = K.reshape(q, shape=(-1, shape[1], self.head_num, key_size))
        k = K.reshape(k, shape=(-1, shape[1], self.head_num, key_size))
        v = K.reshape(v, shape=(-1, shape[1], self.head_num, key_size))

        # attention
        score = tf.einsum('bidh,bjdh->bdij', q, k)  # shape: (batch, head, q_timestep, k_timestep)
        if self.attention_scale:
            score = score / K.sqrt(K.cast(key_size, dtype='float32'))
        # mask padded time-step
        mask = K.expand_dims(mask, axis=1)  # shape: (batch, timestep) -> (batch, 1, k_timestep)
        mask = K.expand_dims(mask, axis=1)  # shape: (batch, 1, 1, k_timestep)
        mask = K.cast(mask, 'int32')
        score = score - 1e10 * K.cast(1-mask, 'float32')  # shape: (batch, head, q_timestep, k_timestep)
        score = K.softmax(score, axis=-1)
        output = tf.einsum('bdij,bjdh->bidh', score, v)  # shape: (batch, q_timestep, head, key_size)

        # reshape back to origin shape
        output = K.reshape(output, shape=shape)

        # dense
        output = self.dense_o(output)
        return output

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape,list):
            return input_shape[0]
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask


class FeedForward(keras.layers.Layer):
    """Position-wise feed-forward layer.

    # Arguments
        units: int >= 0. Dimension of hidden units.
        activation: Activation function to use
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout_rate: 0.0 <= float <= 1.0. Dropout rate for hidden units.

    # Input shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.

    # Output shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.

    # References
        - [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """

    def __init__(self,
                 units,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 dropout_rate=0.0,
                 **kwargs):
        self.supports_masking = True
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.dropout_rate = dropout_rate
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        super(FeedForward, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'dropout_rate': self.dropout_rate,
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.W1 = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W1'.format(self.name),
        )
        if self.use_bias:
            self.b1 = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b1'.format(self.name),
            )
        self.W2 = self.add_weight(
            shape=(self.units, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W2'.format(self.name),
        )
        if self.use_bias:
            self.b2 = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b2'.format(self.name),
            )
        super(FeedForward, self).build(input_shape)

    def call(self, x, mask=None, training=None):
        h = K.dot(x, self.W1)
        if self.use_bias:
            h = K.bias_add(h, self.b1)
        if self.activation is not None:
            h = self.activation(h)
        if 0.0 < self.dropout_rate < 1.0:
            def dropped_inputs():
                return K.dropout(h, self.dropout_rate, K.shape(h))
            h = K.in_train_phase(dropped_inputs, h, training=training)
        y = K.dot(h, self.W2)
        if self.use_bias:
            y = K.bias_add(y, self.b2)
        return y


def get_custom_objects():
    return {
        'gelu': gelu,
        'LayerNormalization': LayerNormalization,
        'MultiHeadAttention': MultiHeadAttention,
        'FeedForward': FeedForward,
        'TrigPosEmbedding': TrigPosEmbedding,
        'EmbeddingRet': EmbeddingRet,
        'EmbeddingSim': EmbeddingSim,
    }


def _wrap_layer(name,
                input_layer,
                build_func,
                dropout_rate=0.0,
                trainable=True):
    """Wrap layers with residual, normalization and dropout.

    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param build_func: A callable that takes the input tensor and generates the output tensor.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    build_output = build_func(input_layer)
    if dropout_rate > 0.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name='%s-Dropout' % name,
        )(build_output)
    else:
        dropout_layer = build_output
    if isinstance(input_layer, list):
        input_layer = input_layer[0]
    add_layer = keras.layers.Add(name='%s-Add' % name)([input_layer, dropout_layer])
    normal_layer = LayerNormalization(
        trainable=trainable,
        name='%s-Norm' % name,
    )(add_layer)
    return normal_layer


def attention_builder(name,
                      head_num,
                      activation,
                      history_only,
                      trainable=True):
    """Get multi-head self-attention builder.

    :param name: Prefix of names for internal layers.
    :param head_num: Number of heads in multi-head self-attention.
    :param activation: Activation for multi-head self-attention.
    :param history_only: Only use history data.
    :param trainable: Whether the layer is trainable.
    :return:
    """
    def _attention_builder(x):
        return MultiHeadAttention(
            head_num=head_num,
            name=name,
        )(x)
    return _attention_builder


def feed_forward_builder(name,
                         hidden_dim,
                         activation,
                         trainable=True):
    """Get position-wise feed-forward layer builder.

    :param name: Prefix of names for internal layers.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param activation: Activation for feed-forward layer.
    :param trainable: Whether the layer is trainable.
    :return:
    """
    def _feed_forward_builder(x):
        return FeedForward(
            units=hidden_dim,
            activation=activation,
            trainable=trainable,
            name=name,
        )(x)
    return _feed_forward_builder


def get_encoder_component(name,
                          input_layer,
                          head_num,
                          hidden_dim,
                          attention_activation=None,
                          feed_forward_activation=gelu,
                          dropout_rate=0.0,
                          trainable=True,):
    """Multi-head self-attention and feed-forward layer.

    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    attention_name = '%s-MultiHeadSelfAttention' % name
    feed_forward_name = '%s-FeedForward' % name
    attention_layer = _wrap_layer(
        name=attention_name,
        input_layer=input_layer,
        build_func=attention_builder(
            name=attention_name,
            head_num=head_num,
            activation=attention_activation,
            history_only=False,
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    feed_forward_layer = _wrap_layer(
        name=feed_forward_name,
        input_layer=attention_layer,
        build_func=feed_forward_builder(
            name=feed_forward_name,
            hidden_dim=hidden_dim,
            activation=feed_forward_activation,
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    return feed_forward_layer


def get_decoder_component(name,
                          input_layer,
                          encoded_layer,
                          head_num,
                          hidden_dim,
                          attention_activation=None,
                          feed_forward_activation=gelu,
                          dropout_rate=0.0,
                          trainable=True):
    """Multi-head self-attention, multi-head query attention and feed-forward layer.

    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param encoded_layer: Encoded layer from encoder.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    self_attention_name = '%s-MultiHeadSelfAttention' % name
    query_attention_name = '%s-MultiHeadQueryAttention' % name
    feed_forward_name = '%s-FeedForward' % name
    self_attention_layer = _wrap_layer(
        name=self_attention_name,
        input_layer=input_layer,
        build_func=attention_builder(
            name=self_attention_name,
            head_num=head_num,
            activation=attention_activation,
            history_only=True,
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    query_attention_layer = _wrap_layer(
        name=query_attention_name,
        input_layer=[self_attention_layer, encoded_layer, encoded_layer],
        build_func=attention_builder(
            name=query_attention_name,
            head_num=head_num,
            activation=attention_activation,
            history_only=False,
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    feed_forward_layer = _wrap_layer(
        name=feed_forward_name,
        input_layer=query_attention_layer,
        build_func=feed_forward_builder(
            name=feed_forward_name,
            hidden_dim=hidden_dim,
            activation=feed_forward_activation,
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    return feed_forward_layer


def get_encoders(encoder_num,
                 input_layer,
                 head_num,
                 hidden_dim,
                 attention_activation=None,
                 feed_forward_activation=gelu,
                 dropout_rate=0.0,
                 trainable=True):
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
    last_layer = input_layer
    for i in range(encoder_num):
        last_layer = get_encoder_component(
            name='Encoder-%d' % (i + 1),
            input_layer=last_layer,
            head_num=head_num,
            hidden_dim=hidden_dim,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
            trainable=trainable,
        )
    return last_layer


def get_decoders(decoder_num,
                 input_layer,
                 encoded_layer,
                 head_num,
                 hidden_dim,
                 attention_activation=None,
                 feed_forward_activation=gelu,
                 dropout_rate=0.0,
                 trainable=True):
    """Get decoders.

    :param decoder_num: Number of decoder components.
    :param input_layer: Input layer.
    :param encoded_layer: Encoded layer from encoder.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    last_layer = input_layer
    for i in range(decoder_num):
        last_layer = get_decoder_component(
            name='Decoder-%d' % (i + 1),
            input_layer=last_layer,
            encoded_layer=encoded_layer,
            head_num=head_num,
            hidden_dim=hidden_dim,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
            trainable=trainable,
        )
    return last_layer


def get_model(token_num,
              embed_dim,
              encoder_num,
              decoder_num,
              head_num,
              hidden_dim,
              attention_activation=None,
              feed_forward_activation=gelu,
              dropout_rate=0.0,
              use_same_embed=True,
              embed_weights=None,
              embed_trainable=None,
              trainable=True):
    """Get full model without compilation.

    :param token_num: Number of distinct tokens.
    :param embed_dim: Dimension of token embedding.
    :param encoder_num: Number of encoder components.
    :param decoder_num: Number of decoder components.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param use_same_embed: Whether to use the same token embedding layer. `token_num`, `embed_weights` and
                           `embed_trainable` should be lists of two elements if it is False.
    :param embed_weights: Initial weights of token embedding.
    :param embed_trainable: Whether the token embedding is trainable. It will automatically set to False if the given
                            value is None when embedding weights has been provided.
    :param trainable: Whether the layers are trainable.
    :return: Keras model.
    """
    if not isinstance(token_num, list):
        token_num = [token_num, token_num]
    encoder_token_num, decoder_token_num = token_num

    if not isinstance(embed_weights, list):
        embed_weights = [embed_weights, embed_weights]
    encoder_embed_weights, decoder_embed_weights = embed_weights
    if encoder_embed_weights is not None:
        encoder_embed_weights = [encoder_embed_weights]
    if decoder_embed_weights is not None:
        decoder_embed_weights = [decoder_embed_weights]

    if not isinstance(embed_trainable, list):
        embed_trainable = [embed_trainable, embed_trainable]
    encoder_embed_trainable, decoder_embed_trainable = embed_trainable
    if encoder_embed_trainable is None:
        encoder_embed_trainable = encoder_embed_weights is None
    if decoder_embed_trainable is None:
        decoder_embed_trainable = decoder_embed_weights is None

    if use_same_embed:
        encoder_embed_layer = decoder_embed_layer = EmbeddingRet(
            input_dim=encoder_token_num,
            output_dim=embed_dim,
            mask_zero=True,
            weights=encoder_embed_weights,
            trainable=encoder_embed_trainable,
            name='Token-Embedding',
        )
    else:
        encoder_embed_layer = EmbeddingRet(
            input_dim=encoder_token_num,
            output_dim=embed_dim,
            mask_zero=True,
            weights=encoder_embed_weights,
            trainable=encoder_embed_trainable,
            name='Encoder-Token-Embedding',
        )
        decoder_embed_layer = EmbeddingRet(
            input_dim=decoder_token_num,
            output_dim=embed_dim,
            mask_zero=True,
            weights=decoder_embed_weights,
            trainable=decoder_embed_trainable,
            name='Decoder-Token-Embedding',
        )
    encoder_input = keras.layers.Input(shape=(None,), name='Encoder-Input')
    encoder_embed = TrigPosEmbedding(
        mode=TrigPosEmbedding.MODE_ADD,
        name='Encoder-Embedding',
    )(encoder_embed_layer(encoder_input)[0])
    encoded_layer = get_encoders(
        encoder_num=encoder_num,
        input_layer=encoder_embed,
        head_num=head_num,
        hidden_dim=hidden_dim,
        attention_activation=attention_activation,
        feed_forward_activation=feed_forward_activation,
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    decoder_input = keras.layers.Input(shape=(None,), name='Decoder-Input')
    decoder_embed, decoder_embed_weights = decoder_embed_layer(decoder_input)
    decoder_embed = TrigPosEmbedding(
        mode=TrigPosEmbedding.MODE_ADD,
        name='Decoder-Embedding',
    )(decoder_embed)
    decoded_layer = get_decoders(
        decoder_num=decoder_num,
        input_layer=decoder_embed,
        encoded_layer=encoded_layer,
        head_num=head_num,
        hidden_dim=hidden_dim,
        attention_activation=attention_activation,
        feed_forward_activation=feed_forward_activation,
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    output_layer = EmbeddingSim(
        trainable=trainable,
        name='Decoder-Output',
    )([decoded_layer, decoder_embed_weights])
    return keras.models.Model(inputs=[encoder_input, decoder_input], outputs=output_layer)


def _get_max_suffix_repeat_times(tokens, max_len):
    detect_len = min(max_len, len(tokens))
    next = [-1] * detect_len
    k = -1
    for i in range(1, detect_len):
        while k >= 0 and tokens[len(tokens) - i - 1] != tokens[len(tokens) - k - 2]:
            k = next[k]
        if tokens[len(tokens) - i - 1] == tokens[len(tokens) - k - 2]:
            k += 1
        next[i] = k
    max_repeat = 1
    for i in range(2, detect_len):
        if next[i] >= 0 and (i + 1) % (i - next[i]) == 0:
            max_repeat = max(max_repeat, (i + 1) // (i - next[i]))
    return max_repeat


def decode(model,
           tokens,
           start_token,
           end_token,
           pad_token,
           top_k=1,
           temperature=1.0,
           max_len=10000,
           max_repeat=10,
           max_repeat_block=10):
    """Decode with the given model and input tokens.

    :param model: The trained model.
    :param tokens: The input tokens of encoder.
    :param start_token: The token that represents the start of a sentence.
    :param end_token: The token that represents the end of a sentence.
    :param pad_token: The token that represents padding.
    :param top_k: Choose the last token from top K.
    :param temperature: Randomness in boltzmann distribution.
    :param max_len: Maximum length of decoded list.
    :param max_repeat: Maximum number of repeating blocks.
    :param max_repeat_block: Maximum length of the repeating block.
    :return: Decoded tokens.
    """
    is_single = not isinstance(tokens[0], list)
    if is_single:
        tokens = [tokens]
    batch_size = len(tokens)
    decoder_inputs = [[start_token] for _ in range(batch_size)]
    outputs = [None for _ in range(batch_size)]
    output_len = 1
    while len(list(filter(lambda x: x is None, outputs))) > 0:
        output_len += 1
        batch_inputs, batch_outputs = [], []
        max_input_len = 0
        index_map = {}
        for i in range(batch_size):
            if outputs[i] is None:
                index_map[len(batch_inputs)] = i
                batch_inputs.append(tokens[i][:])
                batch_outputs.append(decoder_inputs[i])
                max_input_len = max(max_input_len, len(tokens[i]))
        for i in range(len(batch_inputs)):
            batch_inputs[i] += [pad_token] * (max_input_len - len(batch_inputs[i]))
        predicts = model.predict([np.array(batch_inputs), np.array(batch_outputs)])
        for i in range(len(predicts)):
            if top_k == 1:
                last_token = predicts[i][-1].argmax(axis=-1)
            else:
                probs = [(prob, j) for j, prob in enumerate(predicts[i][-1])]
                probs.sort(reverse=True)
                probs = probs[:top_k]
                indices, probs = list(map(lambda x: x[1], probs)), list(map(lambda x: x[0], probs))
                probs = np.array(probs) / temperature
                probs = probs - np.max(probs)
                probs = np.exp(probs)
                probs = probs / np.sum(probs)
                last_token = np.random.choice(indices, p=probs)
            decoder_inputs[index_map[i]].append(last_token)
            if last_token == end_token or\
                    (max_len is not None and output_len >= max_len) or\
                    _get_max_suffix_repeat_times(decoder_inputs[index_map[i]],
                                                 max_repeat * max_repeat_block) >= max_repeat:
                outputs[index_map[i]] = decoder_inputs[index_map[i]]
    if is_single:
        outputs = outputs[0]
    return outputs
