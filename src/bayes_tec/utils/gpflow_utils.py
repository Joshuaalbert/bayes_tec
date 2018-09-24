from gpflow.actions import Action, Loop
from gpflow.training import AdamOptimizer
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

def train_with_adam(model, learning_rate, iterations, callback=None):

    
    with tf.variable_scope("learning_rate"):
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-1
        decay_steps = int(iterations//5)
        decay_rate = 1./2.
        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                  tf.assign_add(global_step,1), decay_steps, decay_rate, staircase=True)
    tf.summary.scalar("optimisation/learning_rate",learning_rate)
    sess = model.enquire_session()
    tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='learning_rate')
    sess.run(tf.variables_initializer(var_list=tf_vars))
#    gp_model.initialize(session=sess)



    
    assert isinstance(callback, (tuple,list))
    for c in callback:
        c.init()
    adam = AdamOptimizer(learning_rate).make_optimize_action(model)
    actions = [adam]
    actions = actions if callback is None else actions + callback

    Loop(actions, stop=iterations)()
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



