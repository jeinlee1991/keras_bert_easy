from .backend import keras, initializers, regularizers, constraints
from .backend import backend as K


class LayerNormalization(keras.layers.Layer):

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer

        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs


class PositionEmbedding(keras.layers.Layer):
    """Turn integers (positions) into dense vectors of fixed size.
    eg. [[-4], [10]] -> [[0.25, 0.1], [0.6, -0.2]]

    Expand mode: negative integers (relative position) could be used in this mode.
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
                 input_dim,
                 output_dim,
                 mode=MODE_EXPAND,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 **kwargs):
        """
        :param input_dim: The maximum absolute value of positions.
        :param output_dim: The embedding dimension.
        :param embeddings_initializer:
        :param embeddings_regularizer:
        :param activity_regularizer:
        :param embeddings_constraint:
        :param mask_zero: The index that represents padding. Only works in `append` mode.
        :param kwargs:
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = mode
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.embeddings_constraint = keras.constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero is not False

        self.embeddings = None
        super(PositionEmbedding, self).__init__(**kwargs)

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'mode': self.mode,
                  'embeddings_initializer': keras.initializers.serialize(self.embeddings_initializer),
                  'embeddings_regularizer': keras.regularizers.serialize(self.embeddings_regularizer),
                  'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
                  'embeddings_constraint': keras.constraints.serialize(self.embeddings_constraint),
                  'mask_zero': self.mask_zero}
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.mode == self.MODE_EXPAND:
            self.embeddings = self.add_weight(
                shape=(self.input_dim * 2 + 1, self.output_dim),
                initializer=self.embeddings_initializer,
                name='embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
            )
        else:
            self.embeddings = self.add_weight(
                shape=(self.input_dim, self.output_dim),
                initializer=self.embeddings_initializer,
                name='embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
            )
        super(PositionEmbedding, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        if self.mode == self.MODE_EXPAND:
            if self.mask_zero:
                output_mask = K.not_equal(inputs, self.mask_zero)
            else:
                output_mask = None
        else:
            output_mask = mask
        return output_mask

    def compute_output_shape(self, input_shape):
        if self.mode == self.MODE_EXPAND:
            return input_shape + (self.output_dim,)
        if self.mode == self.MODE_CONCAT:
            return input_shape[:-1] + (input_shape[-1] + self.output_dim,)
        return input_shape

    def call(self, inputs, **kwargs):
        if self.mode == self.MODE_EXPAND:
            if K.dtype(inputs) != 'int32':
                inputs = K.cast(inputs, 'int32')
            return K.gather(
                self.embeddings,
                K.minimum(K.maximum(inputs, -self.input_dim), self.input_dim) + self.input_dim,
            )
        input_shape = K.shape(inputs)
        if self.mode == self.MODE_ADD:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], input_shape[2]
        else:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], self.output_dim
        pos_embeddings = K.tile(
            K.expand_dims(self.embeddings[:seq_len, :self.output_dim], axis=0),
            [batch_size, 1, 1],
        )
        if self.mode == self.MODE_ADD:
            return inputs + pos_embeddings
        return K.concatenate([inputs, pos_embeddings], axis=-1)


class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return [super(TokenEmbedding, self).compute_output_shape(input_shape), (self.input_dim, self.output_dim)]

    def compute_mask(self, inputs, mask=None):
        return [super(TokenEmbedding, self).compute_mask(inputs, mask), None]

    def call(self, inputs):
        return [super(TokenEmbedding, self).call(inputs), self.embeddings + 0]


def get_embedding(inputs, token_num, pos_num, embed_dim, dropout_rate=0.1, trainable=True):
    """Get embedding layer.

    See: https://arxiv.org/pdf/1810.04805.pdf

    :param inputs: Input layers.
    :param token_num: Number of tokens.
    :param pos_num: Maximum position.
    :param embed_dim: The dimension of all embedding layers.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: The merged embedding layer and weights of token embedding.
    """
    tokenembedding, embed_weights = TokenEmbedding(
        input_dim=token_num,
        output_dim=embed_dim,
        mask_zero=True,
        trainable=trainable,
        name='Embedding-Token',
    )(inputs[0])

    segmentembedding = keras.layers.Embedding(
        input_dim=2,
        output_dim=embed_dim,
        trainable=trainable,
        name='Embedding-Segment',
    )(inputs[1])

    embed_layer = keras.layers.Add(name='Embedding-Token-Segment')([tokenembedding, segmentembedding])

    embed_layer = PositionEmbedding(
        input_dim=pos_num,
        output_dim=embed_dim,
        mode=PositionEmbedding.MODE_ADD,
        trainable=trainable,
        name='Embedding-Position',
    )(embed_layer)
    return embed_layer, embed_weights


class EmbeddingSimilarity(keras.layers.Layer):
    """Calculate similarity between features and token embeddings with bias term."""

    def __init__(self,
                 initializer='zeros',
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        """Initialize the layer.

        :param output_dim: Same as embedding output dimension.
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


class TaskEmbedding(keras.layers.Layer):
    """Embedding for tasks.

        # Arguments
            input_dim: int > 0. Number of the tasks. Plus 1 if `mask_zero` is enabled.
            output_dim: int >= 0. Dimension of the dense embedding.
            embeddings_initializer: Initializer for the `embeddings` matrix.
            embeddings_regularizer: Regularizer function applied to the `embeddings` matrix.
            embeddings_constraint: Constraint function applied to the `embeddings` matrix.
            mask_zero: Generate zeros for 0 index if it is `True`.

        # Input shape
            Previous embeddings, 3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
            Task IDs, 2D tensor with shape: `(batch_size, 1)`.

        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
        """

    def __init__(self,
                 input_dim,
                 output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 **kwargs):
        super(TaskEmbedding, self).__init__(**kwargs)
        self.supports_masking = True
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero

        self.embeddings = None

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            name='embeddings',
        )
        super(TaskEmbedding, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        output_mask = None
        if mask is not None:
            output_mask = mask[0]
        return output_mask

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, **kwargs):
        inputs, tasks = inputs
        if K.dtype(tasks) != 'int32':
            tasks = K.cast(tasks, 'int32')
        task_embed = K.gather(self.embeddings, tasks)
        if self.mask_zero:
            task_embed = task_embed * K.expand_dims(K.cast(K.not_equal(tasks, 0), K.floatx()), axis=-1)
        if K.backend() == 'theano':
            task_embed = K.tile(task_embed, (1, K.shape(inputs)[1], 1))
        return inputs + task_embed

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embeddings_initializer': initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer': regularizers.serialize(self.embeddings_regularizer),
            'embeddings_constraint': constraints.serialize(self.embeddings_constraint),
            'mask_zero': self.mask_zero,
        }
        base_config = super(TaskEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaskedGlobalMaxPool1D(keras.layers.Layer):

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