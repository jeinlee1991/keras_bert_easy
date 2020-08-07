# 分层学习率，特别适用于迁移学习及使用预训练模型的场景。
# 用 BERT / XLNET / ALBERT / RoBERTa / ELECTRA...做fine-tune，
# 使用分层学习率的优化器，往往能得到更好的效果。

import tensorflow as tf
from tensorflow.python.keras.optimizers import Optimizer as tfkerasOptimizer
import keras
import keras.backend as K


def get_weights_by_layernames(model, layernames, use_tfkeras=False):
    if use_tfkeras:
        if not isinstance(model, tf.keras.Model):
            raise ValueError("the input `model` must be a instance of tf.keras.Model")
    else:
        if not isinstance(model, keras.Model):
            raise ValueError("the input `model` must be a instance of keras.Model")

    weights = []
    for name in layernames:
        layer = model.get_layer(name)
        weights += [weight.name for weight in layer.weights]
    return weights


class keras_Adam_2lr(keras.optimizers.Optimizer):
    """支持2段式学习率的Adam optimizer.
        针对keras编写（切勿使用tf.keras），且要求keras.__version__ <= 2.3.1
    # Arguments
        final_layers: list of final layers' weights (layers close to output layer, including output layer)
        lr: float >= 0. List of Learning rates. [Early layers, Final Layers]
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    """

    def __init__(self, final_layers, lr=(1e-5, 1e-4),
                 beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        super(keras_Adam_2lr, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lrate = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')

            self.final_layers = final_layers

        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lrate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        # Setting lr of the initial layers
        lr_grp = lr_t[0]
        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):

            # Updating lr when the split layer is encountered
            if p.name in self.final_layers:
                lr_grp = lr_t[1]

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_grp * m_t / (K.sqrt(vhat_t) + self.epsilon)  # Using updated lr
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_grp * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': (K.get_value(self.lrate)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad}
        base_config = super(keras_Adam_2lr, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def get_weights_by_layernames(model, layernames):
        return get_weights_by_layernames(model, layernames, use_tfkeras=False)


class tfkeras_Adam_2lr(tfkerasOptimizer):
    """支持2段式学习率的Adam optimizer.
        针对tf.keras编写（切勿使用keras）
    # Arguments
        final_layers: list of final layers' weights (layers close to output layer, including output layer)
        lr: float >= 0. List of Learning rates. [Early layers, Final Layers]
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    """

    def __init__(self, final_layers, lr=(1e-5, 1e-4),
                 beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        super(tfkeras_Adam_2lr, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lrate = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')

            self.final_layers = final_layers

        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lrate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        # Setting lr of the initial layers
        lr_grp = lr_t[0]
        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):

            # Updating lr when the split layer is encountered
            if p.name in self.final_layers:
                lr_grp = lr_t[1]

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_grp * m_t / (K.sqrt(vhat_t) + self.epsilon)  # Using updated lr
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_grp * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': (K.get_value(self.lrate)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad}
        base_config = super(tfkeras_Adam_2lr, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def get_weights_by_layernames(model, layernames):
        return get_weights_by_layernames(model, layernames, use_tfkeras=True)


class keras_Adam_3lr(keras.optimizers.Optimizer):
    """支持3段式学习率的Adam optimizer.
        针对keras编写（切勿使用tf.keras），且要求keras.__version__ <= 2.3.1
    # Arguments
        middle_layers: list of middle layers' weights
        final_layers: list of final layers' weights (layers close to output layer, including output layer)
        lr: float >= 0. List of Learning rates. [Early layers, Middle layers, Final Layers]
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    """

    def __init__(self, middle_layers, final_layers, lr=(1e-5, 1e-4, 1e-2),
                 beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        super(keras_Adam_3lr, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lrate = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')

            self.middle_layers = middle_layers
            self.final_layers = final_layers
            # print('middle_layers weights:', middle_layers)
            # print('final_layers weights:', final_layers)
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lrate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        # Setting lr of the initial layers
        lr_grp = lr_t[0]
        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):

            # Updating lr when the split layer is encountered
            if p.name in self.middle_layers:
                lr_grp = lr_t[1]
            if p.name in self.final_layers:
                lr_grp = lr_t[2]

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_grp * m_t / (K.sqrt(vhat_t) + self.epsilon)  # Using updated lr
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_grp * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': (K.get_value(self.lrate)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad}
        base_config = super(keras_Adam_3lr, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def get_weights_by_layernames(model, layernames):
        return get_weights_by_layernames(model, layernames, use_tfkeras=False)


class tfkeras_Adam_3lr(tfkerasOptimizer):
    """支持3段式学习率的Adam optimizer.
        针对tf.keras编写（切勿使用keras）
    # Arguments
        middle_layers: list of middle layers' weights
        final_layers: list of final layers' weights (layers close to output layer, including output layer)
        lr: float >= 0. List of Learning rates. [Early layers, Middle layers, Final Layers]
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    """

    def __init__(self, middle_layers, final_layers, lr=(1e-5, 1e-4, 1e-2),
                 beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        super(tfkeras_Adam_3lr, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lrate = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')

            self.middle_layers = middle_layers
            self.final_layers = final_layers
            # print('middle_layers weights:', middle_layers)
            # print('final_layers weights:', final_layers)
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lrate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        # Setting lr of the initial layers
        lr_grp = lr_t[0]
        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            # print('params name: ', p.name)
            # Updating lr when the split layer is encountered
            if p.name in self.middle_layers:
                lr_grp = lr_t[1]
            if p.name in self.final_layers:
                lr_grp = lr_t[2]

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_grp * m_t / (K.sqrt(vhat_t) + self.epsilon)  # Using updated lr
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_grp * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': (K.get_value(self.lrate)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad}
        base_config = super(tfkeras_Adam_3lr, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def get_weights_by_layernames(model, layernames):
        return get_weights_by_layernames(model, layernames, use_tfkeras=True)


class keras_Adam_lr_decay(keras.optimizers.Optimizer):
    """Adam optimizer that supports layer-wise-decayed learning rate.
       Layer-wise-decayed learning rate means that the learning rate decays
            by the same power from output-layer to input-layer.
       For use of keras API (don't use tf.keras here)，requiring keras.__version__ <= 2.3.1
    # Arguments
        weight_name_to_layer_depth: dict, get the layer depth according to weight name.
            With its layer depth, the weight can be bound with layer-wise-decayed learning rate.
            Output layer gets the biggest depth.
        n_layers: int, layers of the model (eg. n_layers = len(model.layers))
        layerwise_lr_decay_power: float, 0 < layerwise_lr_decay_power <= 1.
            lr of depth m, i.e. lr_m = lr * pow(layerwise_lr_decay_power, (n_layers - m))
        lr: float >= 0.  Learning rates.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    """

    def __init__(self, weight_name_to_layer_depth, n_layers, layerwise_lr_decay_power=0.8,
                 lr=1e-3, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        super(keras_Adam_lr_decay, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lrate = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')

        self.weight_name_to_layer_depth = weight_name_to_layer_depth
        self.n_layers = n_layers
        self.layerwise_lr_decay_power = layerwise_lr_decay_power
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lrate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        # Setting lr of the initial layers
        lr_grp = lr_t
        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            # Updating lr according to layer depth (input_layer_depth = 1)
            depth = self.weight_name_to_layer_depth.get(p.name, self.n_layers)
            if 1 <= depth <= self.n_layers:
                lr_grp = lr_t * K.pow(self.layerwise_lr_decay_power, self.n_layers-depth)
                # print('param name: %s, depth: %d, lr: %s'%(p.name, depth, lr_grp))
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_grp * m_t / (K.sqrt(vhat_t) + self.epsilon)  # Using updated lr
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_grp * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': (K.get_value(self.lrate)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad}
        base_config = super(keras_Adam_lr_decay, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def build_dict_layer_depth(keras_model):
        if not isinstance(keras_model, keras.Model):
            raise ValueError("the input `keras_model` must be a instance of keras.Model")

        weight_name_to_layer_depth = {}
        for i, layer in enumerate(keras_model.layers):
            print('layer name: ', layer.name)
            depth = i + 1
            for w in layer.weights:
                print('\t weights name: ', w.name)
                weight_name_to_layer_depth[w.name] = depth
        return weight_name_to_layer_depth


class tfkeras_Adam_lr_decay(tfkerasOptimizer):
    """Adam optimizer that supports layer-wise-decayed learning rate.
       Layer-wise-decayed learning rate means that the learning rate decays
            by the same power from output-layer to input-layer.
       For use of tf.keras API (don't use keras here)
    # Arguments
        weight_name_to_layer_depth: dict, get the layer depth according to weight name.
            With its layer depth, the weight can be bound with layer-wise-decayed learning rate.
            Output layer gets the biggest depth.
        n_layers: int, layers of the model (eg. n_layers = len(model.layers))
        layerwise_lr_decay_power: float, 0 < layerwise_lr_decay_power <= 1.
            lr of depth m, i.e. lr_m = lr * pow(layerwise_lr_decay_power, (n_layers - m))
        lr: float >= 0.  Learning rates.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    """

    def __init__(self, weight_name_to_layer_depth, n_layers, layerwise_lr_decay_power=0.8,
                 lr=1e-3, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        super(tfkeras_Adam_lr_decay, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lrate = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')

        self.weight_name_to_layer_depth = weight_name_to_layer_depth
        self.n_layers = n_layers
        self.layerwise_lr_decay_power = layerwise_lr_decay_power
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lrate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        # Setting lr of the initial layers
        lr_grp = lr_t
        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            # Updating lr according to layer depth (input_layer_depth = 1)
            depth = self.weight_name_to_layer_depth.get(p.name, self.n_layers)
            if 1 <= depth <= self.n_layers:
                lr_grp = lr_t * K.pow(self.layerwise_lr_decay_power, self.n_layers-depth)
                # print('param name: %s, depth: %d, lr: %s'%(p.name, depth, lr_grp))
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_grp * m_t / (K.sqrt(vhat_t) + self.epsilon)  # Using updated lr
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_grp * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': (K.get_value(self.lrate)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad}
        base_config = super(tfkeras_Adam_lr_decay, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def build_dict_layer_depth(tfkeras_model):
        if not isinstance(tfkeras_model, tf.keras.Model):
            raise ValueError("the input `tfkeras_model` must be a instance of tf.keras.Model")

        weight_name_to_layer_depth = {}
        for i, layer in enumerate(tfkeras_model.layers):
            print('layer name: ', layer.name)
            depth = i + 1
            for w in layer.weights:
                print('\t weights name: ', w.name)
                weight_name_to_layer_depth[w.name] = depth
        return weight_name_to_layer_depth
