from gpflow.actions import Action, Loop
from gpflow.training import NatGradOptimizer, AdamOptimizer, ScipyOptimizer
from gpflow import settings
from gpflow.transforms import Transform
from ..logging import logging
import tensorflow as tf
import os
import numpy as np

class PrintAction(Action):
    def __init__(self, model, text):
        self.model = model
        self.text = text
        
    def run(self, ctx):
        likelihood = ctx.session.run(self.model.likelihood_tensor)
        logging.warning('{}: iteration {} likelihood {:.4f}'.format(self.text, ctx.iteration, likelihood))

class SendSummary(Action):
    def __init__(self, model, writer, write_period=10):
        self.write_period = write_period
        self.iteration = 0
        self.model = model
        self.writer = writer


    def init(self):
        parameters = list(self.model.parameters)

        other_summaries = tf.summary.merge_all()
        if other_summaries is None:
            other_summaries = []
        else:
            if not isinstance(other_summaries, (list,tuple)):
                other_summaries = [other_summaries]
            other_summaries = list(other_summaries)
        

        # Add scalar parameters
        scalar_summaries = [tf.summary.scalar(p.pathname, tf.reshape(p.constrained_tensor, []))
                          for p in parameters if (p.size == 1 and p.trainable)]

        scalar_summaries.append(tf.summary.scalar("optimisation/likelihood",
                                               self.model._likelihood_tensor))
        self.scalar_summary = tf.summary.merge(scalar_summaries)

        # Add non-scalar parameters
#        self.hist_summary = tf.summary.merge([
#            tf.summary.histogram('q_mu',model.q_mu.constrained_tensor), 
#            tf.summary.histogram('q_sqrt',model.q_sqrt.unconstrained_tensor)
#            ])
        hist_summaries = [tf.summary.histogram(p.pathname, p.constrained_tensor)
                          for p in parameters if p.size > 1]
        self.hist_summary = tf.summary.merge(hist_summaries)

        self.summary = tf.summary.merge([self.scalar_summary,self.hist_summary] + other_summaries)

        
    def run(self, ctx):
        if self.iteration % self.write_period == 0:
            summary = ctx.session.run(self.summary)
            self.writer.add_summary(summary,global_step=ctx.iteration)
        self.iteration += 1

class SaveModel(Action):
    def __init__(self, checkpoint_dir, save_period=1000):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        os.makedirs(self.checkpoint_dir,exist_ok=True)

        self.save_period = save_period
        self.iteration = 0
        self.saver = tf.train.Saver(max_to_keep=1)
        
    def run(self, ctx):
        if self.iteration % self.save_period == 0:
            self.saver.save(ctx.session, self.checkpoint_dir,
                         global_step=ctx.iteration)
        self.iteration += 1

def restore_session(session, checkpoint_dir):
    """
    Restores Tensorflow session from the latest checkpoint.
    :param session: The TF session
    :param checkpoint_dir: checkpoint files directory.
    """
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    logger = settings.logger()
    if logger.isEnabledFor(logging.INFO):
        logger.info("Restoring session from `%s`.", checkpoint_path)

    saver = tf.train.Saver(max_to_keep=1)
    saver.restore(session, checkpoint_path)

def train_with_adam(model, iterations, callback=None, **kwargs):
    #,initial_learning_rate=0.03,learning_rate_steps=2.3,
    #          learning_rate_decay=1.5,


    with tf.variable_scope("learning_rate"):
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.03
        decay_steps = int(iterations/2.)
        decay_rate = 1./1.5
        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                  tf.assign_add(global_step,1), decay_steps, decay_rate, staircase=True)
    tf.summary.scalar("optimisation/learning_rate",learning_rate)
    sess = model.enquire_session()
    tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='learning_rate')
    sess.run(tf.variables_initializer(var_list=tf_vars))
        
    assert isinstance(callback, (tuple,list))
    adam = AdamOptimizer(learning_rate).make_optimize_action(model)
    actions = [adam]
    actions = actions if callback is None else actions + callback
    for c in callback:
        try:
            c.init()
        except:
            pass


    Loop(actions, stop=iterations)()
    model.anchor(model.enquire_session())


def train_with_bfgs(model, learning_rate, iterations, callback=None):

    
    sess = model.enquire_session()
    
    assert isinstance(callback, (tuple,list))
    for c in callback:
        c.init()
    adam = ScipyOptimizer().make_optimize_action(model)
    actions = [adam]
    actions = actions if callback is None else actions + callback

    Loop(actions)()
    model.anchor(model.enquire_session())


class GammaSchedule(Action):
    def __init__(self, op_increment_gamma):
        self.op_increment_gamma = op_increment_gamma

    def run(self, ctx):
        ctx.session.run(self.op_increment_gamma)

def train_with_nat_and_adam(model, initial_learning_rate=0.03,learning_rate_steps=2,
              learning_rate_decay=1.5,gamma_start=1e-5,gamma_add=1e-3,gamma_mul=1.1,
             gamma_max=0.1,gamma_fallback=1e-1,iterations=500, var_list=None, callback=None, **kwargs):
    # we'll make use of this later when we use a XiTransform
    if var_list is None:
        var_list = [[model.q_mu, model.q_sqrt]]

    # we don't want adam optimizing these
    model.q_mu.set_trainable(False)
    model.q_sqrt.set_trainable(False)

    with tf.variable_scope("learning_rate"):
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = initial_learning_rate
        decay_steps = int(iterations/learning_rate_steps)
        decay_rate = 1./learning_rate_decay
        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                  tf.assign_add(global_step,1), decay_steps, decay_rate, staircase=True)
    tf.summary.scalar("optimisation/learning_rate",learning_rate)
    sess = model.enquire_session()
    tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='learning_rate')
    sess.run(tf.variables_initializer(var_list=tf_vars))

    
    with tf.variable_scope("gamma"):

#        gamma = tf.Variable(gamma_start, dtype=tf.float64)
#        beta = tf.Variable(1.,dtype=tf.float64)
        
        gamma_start = tf.cast(gamma_start,tf.float64)
        gamma_max = tf.cast(gamma_max,tf.float64)
        mul_step = tf.cast(gamma_mul,tf.float64)
        add_step = tf.cast(gamma_add,tf.float64)
        gamma = tf.Variable(gamma_start, dtype=tf.float64)

        gamma_ref = tf.identity(gamma)
        

        gamma_fallback = tf.cast(gamma_fallback, tf.float64)   # we'll reduce by this factor if there's a cholesky failure 
        op_fallback_gamma = tf.assign(gamma, gamma * gamma_fallback) 
        diff = tf.where(gamma_ref*mul_step < add_step, gamma_ref*mul_step, add_step)
        op_gamma_inc = tf.assign(gamma, tf.where(gamma_ref + diff > gamma_max, gamma_max, gamma_ref + diff))

    tf.summary.scalar("optimisation/gamma",gamma)
    sess = model.enquire_session()
    tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gamma')
    sess.run(tf.variables_initializer(var_list=tf_vars))

    natgrad = NatGradOptimizer(gamma_ref).make_optimize_action(model, var_list=var_list)
    adam = AdamOptimizer(learning_rate).make_optimize_action(model)

    actions = [adam, natgrad, GammaSchedule(op_gamma_inc)]
    actions = actions if callback is None else actions + callback
    for c in callback:
        try:
            c.init()
        except:
            pass

    sess = model.enquire_session()
    it = 0
    while it < iterations:
        try:
            looper = Loop(actions, start=it, stop=iterations)
            looper()
            it = looper.iteration
        except tf.errors.InvalidArgumentError:
            it = looper.iteration
            g, gf = sess.run([gamma_ref, op_fallback_gamma])
            logging.info('gamma = {} on iteration {} is too big! Falling back to {}'.format(g, it, gf))
            
    model.anchor(model.enquire_session())

def train_with_nat(model, gamma_start=1e-5,gamma_add=1e-3,gamma_mul=1.04,
             gamma_max=0.1,gamma_fallback=1e-1,iterations=500, var_list=None, callback=None, **kwargs):
    # we'll make use of this later when we use a XiTransform
    if var_list is None:
        var_list = [[model.q_mu, model.q_sqrt]]

        
    with tf.variable_scope("gamma"):

        
        gamma_start = tf.cast(gamma_start,tf.float64)
        gamma_max = tf.cast(gamma_max,tf.float64)
        mul_step = tf.cast(gamma_mul,tf.float64)
        add_step = tf.cast(gamma_add,tf.float64)
        gamma = tf.Variable(gamma_start, dtype=tf.float64, trainable=False)

        gamma_ref = tf.identity(gamma)
        

        gamma_fallback = tf.cast(gamma_fallback, tf.float64)   # we'll reduce by this factor if there's a cholesky failure 
        op_fallback_gamma = tf.assign(gamma, gamma_ref * gamma_fallback) 
        diff = tf.where(gamma_ref*mul_step < add_step, gamma_ref*mul_step, add_step)
        op_gamma_inc = tf.assign(gamma, tf.where(gamma_ref + diff > gamma_max, gamma_max, gamma_ref + diff))

    tf.summary.scalar("optimisation/gamma",gamma)
    sess = model.enquire_session()
    tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gamma')
    sess.run(tf.variables_initializer(var_list=tf_vars))

    natgrad = NatGradOptimizer(gamma_ref).make_optimize_action(model, var_list=var_list)

    actions = [natgrad, GammaSchedule(op_gamma_inc)]
    actions = actions if callback is None else actions + callback

    for c in callback:
        try:
            c.init()
        except:
            pass

   
    sess = model.enquire_session()
    it = 0
    while it < iterations:
        try:
            looper = Loop(actions, start=it, stop=iterations)
            looper()
            it = looper.iteration
        except tf.errors.InvalidArgumentError:
            it = looper.iteration
            g,gf = sess.run([gamma_ref, op_fallback_gamma])
            logging.info('gamma = {} on iteration {} is too big! Falling back to {}'.format(g, it, gf))
            
    model.anchor(model.enquire_session())


class Reshape(Transform):
    """
    The exponential transform:
       y = x
       y is of shape y_shape, x is of shape x_shape and are compatible
    """
    def __init__(self, y_shape, x_shape):
        self._xshape = x_shape
        self._yshape = y_shape

    def forward_tensor(self, x):
        return tf.reshape(x, self._yshape)

    def backward_tensor(self, y):
        return tf.reshape(y, self._xshape)

    def forward(self, x):
        return np.reshape(x,self._yshape)

    def backward(self, y):
        return np.reshape(y,self._xshape)

    def log_jacobian_tensor(self, x):
        return tf.zeros((1,), settings.float_type)

    def __str__(self):
        return 'Reshape'


class MatrixSquare(Transform):
    """
    The exponential transform:
       y = x
       y is of shape y_shape, x is of shape x_shape and are compatible
    """
    def __init__(self):
        pass
    def forward_tensor(self, x):
        return tf.matmul(x,x,transpose_b=True)

    def backward_tensor(self, y):
        return tf.cholesky(y)

    def forward(self, x):
        return np.einsum("bij,bkj->bik",x,x)

    def backward(self, y):
        return np.stack([np.linalg.cholesky(yi) for yi in y],axis=0)

    def log_jacobian_tensor(self, x):
        """
        Input (N,L,L)
        """
        return tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))

    def __str__(self):
        return 'MatrixSquare'


class RescaleVec(Transform):
    """
    A transform that can linearly rescale parameters:
    .. math::
       y = factor * x
    Use `Chain` to combine this with another transform such as Log1pe:
    `Chain(Rescale(), otherTransform())` results in
       y = factor * t(x)
    `Chain(otherTransform(), Rescale())` results in
       y = t(factor * x)
    This is useful for avoiding overly large or small scales in optimization/MCMC.
    If you want a transform for a positive quantity of a given scale, you want
    >>> Rescale(scale)(positive)
    """
    def __init__(self, factor=1.0):
        self.factor = np.array(factor)

    def forward_tensor(self, x):
        return x * self.factor

    def forward(self, x):
        return x * self.factor

    def backward_tensor(self, y):
        return y / self.factor

    def backward(self, y):
        return y / self.factor

    def log_jacobian_tensor(self, x):
        factor = tf.cast(self.factor, dtype=settings.float_type)
        log_factor = tf.log(factor)
        return tf.reduce_sum(log_factor)

    def __str__(self):
        return "{}*".format(self.factor)

class LogisticVec(Transform):
    """
    The logistic transform, useful for keeping variables constrained between the limits a and b:
    .. math::
       y = a + (b-a) s(x)
       s(x) = 1 / (1 + \exp(-x))
    """
    def __init__(self, a, b):
        self.a, self.b = np.array(a), np.array(b)

    def forward_tensor(self, x):
        ex = tf.exp(-x)
        return self.a + (self.b - self.a) / (1. + ex)

    def forward(self, x):
        ex = np.exp(-x)
        return self.a + (self.b - self.a) / (1. + ex)

    def backward_tensor(self, y):
        return -tf.log((self.b - self.a) / (y - self.a) - 1.)

    def backward(self, y):
        return -np.log((self.b - self.a) / (y - self.a) - 1.)

    def log_jacobian_tensor(self, x):
        return tf.reduce_sum(x - 2. * tf.log(tf.exp(x) + 1.) + np.log(self.b - self.a))

    def __str__(self):
        return "[{}, {}]".format(self.a, self.b)
