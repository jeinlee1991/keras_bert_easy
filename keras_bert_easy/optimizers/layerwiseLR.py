# 分层学习率，特别适用于迁移学习及使用预训练模型的场景。
# 用 BERT / XLNET / ALBERT / RoBERTa / ELECTRA...做fine-tune，
# 使用分层学习率的优化器，往往能得到更好的效果。

import os
import re
from distutils.util import strtobool
import tensorflow as tf

TF_KERAS = strtobool(os.environ.get('TF_KERAS', '0'))
if TF_KERAS:
    keras = tf.keras
    from tensorflow.python.keras.optimizers import Optimizer as tfkerasOptimizer
    BaseOptimizer = tfkerasOptimizer
else:
    import keras
    BaseOptimizer = keras.optimizers.Optimizer
K = keras.backend


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


def keras_mode_hint():
    if TF_KERAS:
        print('using tf.keras API. tf version: %s' % tf.__version__)
    else:
        version = keras.__version__
        if version > '2.3.1':
            raise ImportError('keras version should not be greater than 2.3.1, got %s' % version)
        print('using keras API. keras version: %s' % version)


def do_use_weight_decay(param_name, exclude_from_weight_decay):
    """Whether to use L2 weight decay for `param_name`."""
    if exclude_from_weight_decay:
        for r in exclude_from_weight_decay:
            if re.search(r, param_name) is not None:
                return False
    return True


class KerasAdam2lr(BaseOptimizer):
    """支持2段式学习率的Adam optimizer.
    # Arguments
        final_layers: list of final layers' weights (layers close to output layer, including output layer)
        lr: float >= 0. List of Learning rates. [Early layers, Final Layers]
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    """

    def __init__(self, final_layers, lr=(1e-5, 1e-4),
                 beta_1=0.9, beta_2=0.999,
                 weight_decay=0, # 0.01
                 exclude_from_weight_decay=("LayerNorm", "layer_norm", "bias"),
                 amsgrad=False, **kwargs):
        super(KerasAdam2lr, self).__init__(**kwargs)

        keras_mode_hint()
        if weight_decay > 0:
            print('using optimizer AdamW...')

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lrate = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')

        self.final_layers = final_layers
        self.weight_decay = weight_decay
        self.exclude_from_weight_decay = exclude_from_weight_decay

        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lrate
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

            # weight decay (L2 regularization)
            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self.weight_decay > 0:
                if do_use_weight_decay(p.name, self.exclude_from_weight_decay):
                    p_t = p_t - lr_grp * self.weight_decay * p

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
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'weight_decay': self.weight_decay,
            'amsgrad': self.amsgrad}
        base_config = super(KerasAdam2lr, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def get_weights_by_layernames(model, layernames):
        return get_weights_by_layernames(model, layernames, use_tfkeras=False)


class KerasAdam3lr(BaseOptimizer):
    """支持3段式学习率的Adam optimizer.
    # Arguments
        middle_layers: list of middle layers' weights
        final_layers: list of final layers' weights (layers close to output layer, including output layer)
        lr: float >= 0. List of Learning rates. [Early layers, Middle layers, Final Layers]
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    """

    def __init__(self, middle_layers, final_layers, lr=(1e-5, 1e-4, 1e-2),
                 beta_1=0.9, beta_2=0.999,
                 weight_decay=0,  # 0.01
                 exclude_from_weight_decay=("LayerNorm", "layer_norm", "bias"),
                 amsgrad=False, **kwargs):
        super(KerasAdam3lr, self).__init__(**kwargs)

        keras_mode_hint()
        if weight_decay > 0:
            print('using optimizer AdamW...')

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lrate = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')

        self.middle_layers = middle_layers
        self.final_layers = final_layers
        self.weight_decay = weight_decay
        self.exclude_from_weight_decay = exclude_from_weight_decay

        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lrate
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

            # weight decay (L2 regularization)
            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self.weight_decay > 0:
                if do_use_weight_decay(p.name, self.exclude_from_weight_decay):
                    p_t = p_t - lr_grp * self.weight_decay * p

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
            # 'lr': (K.get_value(self.lrate)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'weight_decay': self.weight_decay,
            'amsgrad': self.amsgrad}
        base_config = super(KerasAdam3lr, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def get_weights_by_layernames(model, layernames):
        return get_weights_by_layernames(model, layernames, use_tfkeras=False)


class KerasAdamLrDecay(BaseOptimizer):
    """Adam optimizer that supports layer-wise-decayed learning rate.
       Layer-wise-decayed learning rate means that the learning rate decays
            by the same power from output-layer to input-layer.

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
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    """

    def __init__(self, weight_name_to_layer_depth, n_layers,
                 layerwise_lr_decay_power=0.8, lr=1e-3,
                 beta_1=0.9, beta_2=0.999,
                 weight_decay=0,  # 0.01
                 exclude_from_weight_decay=("LayerNorm", "layer_norm", "bias"),
                 amsgrad=False, **kwargs):
        super(KerasAdamLrDecay, self).__init__(**kwargs)

        keras_mode_hint()
        if weight_decay > 0:
            print('using optimizer AdamW...')

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lrate = K.variable(lr, name='lr')
            # print('lr shape:', self.lrate.shape, len(self.lrate.shape))
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')

        self.weight_name_to_layer_depth = weight_name_to_layer_depth
        self.n_layers = n_layers
        self.layerwise_lr_decay_power = layerwise_lr_decay_power
        self.weight_decay = weight_decay
        self.exclude_from_weight_decay = exclude_from_weight_decay

        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lrate
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

            # weight decay (L2 regularization)
            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self.weight_decay > 0:
                if do_use_weight_decay(p.name, self.exclude_from_weight_decay):
                    p_t = p_t - lr_grp * self.weight_decay * p

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
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'weight_decay': self.weight_decay,
            'amsgrad': self.amsgrad}
        base_config = super(KerasAdamLrDecay, self).get_config()
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


class AccumOptimizer(BaseOptimizer):
    """继承Optimizer类，包装原有优化器，实现梯度累积。
    # 参数
        optimizer：优化器实例，支持目前所有的keras优化器；
        steps_per_update：累积的步数。
    # 返回
        一个新的keras优化器
    Inheriting Optimizer class, wrapping the original optimizer
    to achieve a new corresponding optimizer of gradient accumulation.
    # Arguments
        optimizer: an instance of keras optimizer (supporting
                    all keras optimizers currently available);
        steps_per_update: the steps of gradient accumulation
    # Returns
        a new keras optimizer.
    """
    def __init__(self, optimizer, steps_per_update=1, **kwargs):
        super(AccumOptimizer, self).__init__(**kwargs)
        self.optimizer = optimizer
        with K.name_scope(self.__class__.__name__):
            self.steps_per_update = steps_per_update
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.cond = K.equal(self.iterations % self.steps_per_update, 0)
            self.lrate = self.optimizer.lrate

            if len(self.lrate.shape):
                lr_size = self.lrate.shape[0]
                self.optimizer.lrate = K.switch(self.cond, self.optimizer.lrate, K.variable([0.]*lr_size))
            else:
                self.optimizer.lrate = K.switch(self.cond, self.optimizer.lrate, K.variable(0.))

            for attr in ['momentum', 'rho', 'beta_1', 'beta_2']:
                if hasattr(self.optimizer, attr):
                    value = getattr(self.optimizer, attr)
                    setattr(self, attr, value)
                    setattr(self.optimizer, attr, K.switch(self.cond, value, 1 - 1e-7))
            for attr in self.optimizer.get_config():
                if not hasattr(self, attr):
                    value = getattr(self.optimizer, attr)
                    setattr(self, attr, value)
            # 覆盖原有的获取梯度方法，指向累积梯度
            # Cover the original get_gradients method with accumulative gradients.
            def get_gradients(loss, params):
                return [ag / self.steps_per_update for ag in self.accum_grads]
            self.optimizer.get_gradients = get_gradients

    def get_updates(self, loss, params):
        self.updates = [
            K.update_add(self.iterations, 1),
            K.update_add(self.optimizer.iterations, K.cast(self.cond, 'int64')),
        ]
        # 累积梯度 (gradient accumulation)
        grads = self.get_gradients(loss, params)
        self.accum_grads = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        for g, ag in zip(grads, self.accum_grads):
            ag_new = ag + g
            g = g + K.zeros(K.int_shape(ag_new), dtype=K.dtype(ag_new))
            self.updates.append(K.update(ag, K.switch(self.cond, g, ag_new)))
        # 继承optimizer的更新 (inheriting updates of original optimizer)
        self.updates.extend(self.optimizer.get_updates(loss, params)[1:])
        self.weights.extend(self.optimizer.weights)
        return self.updates

    def get_config(self):
        iterations = K.eval(self.iterations)
        K.set_value(self.iterations, 0)
        config = self.optimizer.get_config()
        K.set_value(self.iterations, iterations)
        return config