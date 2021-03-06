import tensorflow as tf
from .backend import keras, K

Layer = keras.layers.Layer


class TrigPosEmbedding(Layer):
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


class EmbeddingSim(Layer):
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


class MultiHeadAttention(Layer):
    """多头注意力机制
    """
    def __init__(
        self,
        head_num,
        use_bias=True,
        attention_prob_dropout_rate=0.,
        attention_scale=True,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.head_num = head_num
        self.use_bias = use_bias
        self.attention_prob_dropout_rate = attention_prob_dropout_rate
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

        # dropout
        score_dropout = keras.layers.Dropout(rate=self.attention_prob_dropout_rate)(score)

        # attention output
        output = tf.einsum('bdij,bjdh->bidh', score_dropout, v)  # shape: (batch, q_timestep, head, key_size)

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


class FeedForward(Layer):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        if isinstance(activation, str):
            self.activation = keras.activations.get(activation)
        else:
            self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        self.dense0 = keras.layers.Dense(
            self.units,
            activation=self.activation,
            use_bias=self.use_bias
        )
        self.dense1 = keras.layers.Dense(
            input_shape[-1],
            use_bias=self.use_bias
        )

    def call(self, inputs, **kwargs):
        x = self.dense0(inputs)
        x = self.dense1(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask


class LayerNormalization(Layer):
    def __init__(self, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        super(LayerNormalization, self).build(input_shape)
        self.gamma = self.add_weight(shape=(hidden_dim,), initializer='ones')
        self.beta = self.add_weight(shape=(hidden_dim,), initializer='zeros')

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + K.epsilon()*K.epsilon())
        x = (inputs - mean) / std
        x = x * self.gamma + self.beta
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask


class PositionEmbedding(Layer):
    def __init__(self, max_pos_num=512, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.max_pos_num = max_pos_num

    def build(self, input_shape):
        hidden_dim = input_shape[2]
        self.pos_emb_layer = keras.layers.Embedding(
            self.max_pos_num, hidden_dim,
            name='embeddings'
        )

    def call(self, inputs, **kwargs):
        shape = K.shape(inputs)
        batch, seqlen = shape[0], shape[1]
        x_pos = K.expand_dims(K.arange(0, seqlen), axis=0)
        x_pos = K.tile(x_pos, [batch,1])
        x_pos_emb = self.pos_emb_layer(x_pos)
        x = keras.layers.Add()([inputs, x_pos_emb])
        return x

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape


class TokenEmbedding(keras.layers.Embedding):
    def call(self, inputs):
        return [super(TokenEmbedding, self).call(inputs),
                self.embeddings]

    def compute_output_shape(self, input_shape):
        return [super(TokenEmbedding, self).compute_output_shape(input_shape),
                K.shape(self.embeddings)]

    def compute_mask(self, inputs, mask=None):
        return [super(TokenEmbedding, self).compute_mask(inputs), None]


class EmbeddingSimilarity(Layer):
    """Calculate similarity between features and token embeddings with bias term."""

    def __init__(self,
                 initializer='zeros',
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        """Initialize the layer.

        :param initializer: Initializer for bias.
        :param regularizer: Regularizer for bias.
        :param constraint: Constraint for bias.
        :param kwargs: Arguments for parent class.
        """
        super(EmbeddingSimilarity, self).__init__(**kwargs)
        self.supports_masking = True
        self.initializer = keras.initializers.get(initializer)
        self.regularizer = keras.regularizers.get(regularizer)
        self.constraint = keras.constraints.get(constraint)
        self.bias = None

    def get_config(self):
        config = {
            'initializer': keras.initializers.serialize(self.initializer),
            'regularizer': keras.regularizers.serialize(self.regularizer),
            'constraint': keras.constraints.serialize(self.constraint),
        }
        base_config = super(EmbeddingSimilarity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.bias = self.add_weight(
            shape=(int(input_shape[1][0]),),  # vocab size
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint,
            name='bias',
        )
        super(EmbeddingSimilarity, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + (input_shape[1][0],)

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def call(self, inputs, mask=None, **kwargs):
        inputs, embeddings = inputs
        outputs = K.bias_add(K.dot(inputs, K.transpose(embeddings)), self.bias)
        return keras.activations.softmax(outputs)


class MaskedGlobalMaxPool1D(Layer):

    def __init__(self, **kwargs):
        super(MaskedGlobalMaxPool1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + (input_shape[-1],)

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs -= K.expand_dims((1.0 - mask) * 1e6, axis=-1)
        return K.max(inputs, axis=-2)


class MaskedConv1D(keras.layers.Conv1D):

    def __init__(self, **kwargs):
        super(MaskedConv1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None and self.padding == 'valid':
            mask = mask[:, self.kernel_size[0] // 2 * self.dilation_rate[0] * 2:]
        return mask

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs *= K.expand_dims(mask, axis=-1)
        return super(MaskedConv1D, self).call(inputs)


class Masked(Layer):
    """Generate output mask based on the given mask.

    The inputs for the layer is the original input layer and the masked locations.

    See: https://arxiv.org/pdf/1810.04805.pdf
    """

    def __init__(self,
                 return_masked=False,
                 **kwargs):
        """Initialize the layer.

        :param return_masked: Whether to return the merged mask.
        :param kwargs: Arguments for parent class.
        """
        super(Masked, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_masked = return_masked

    def get_config(self):
        config = {'return_masked': self.return_masked,}
        base_config = super(Masked, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if self.return_masked:
            return [input_shape[0], input_shape[0][:-1]]
        return input_shape[0]

    def compute_mask(self, inputs, mask=None):
        token_mask = K.not_equal(inputs[1], 0)
        masked = K.all(K.stack([token_mask, mask[0]], axis=0), axis=0)
        if self.return_masked:
            return [masked, None]
        return masked

    def call(self, inputs, mask=None, **kwargs):
        output = inputs[0] + 0
        if self.return_masked:
            return [output, K.cast(self.compute_mask(inputs, mask)[0], K.floatx())]
        return output

