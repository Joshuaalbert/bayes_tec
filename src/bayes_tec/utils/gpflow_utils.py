from gpflow.actions import Action, Loop
from gpflow.training import AdamOptimizer
from gpflow import settings
from ..logging import logging
import tensorflow as tf
import os

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
        parameters = list(self.model.parameters)

        # Add scalar parameters
        scalar_summaries = [tf.summary.scalar(p.pathname, tf.reshape(p.constrained_tensor, []))
                          for p in parameters if p.size == 1]
        scalar_summaries.append(tf.summary.scalar("optimisation/likelihood",
                                               self.model._likelihood_tensor))
        self.scalar_summary = tf.summary.merge(scalar_summaries)

        # Add non-scalar parameters
        hist_summaries = [tf.summary.histogram(p.pathname, p.constrained_tensor)
                          for p in parameters if p.size > 1]
        self.hist_summary = tf.summary.merge(hist_summaries)

        self.summary = tf.summary.merge([self.scalar_summary,self.hist_summary])

        
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
    assert isinstance(callback, (tuple,list))
    adam = AdamOptimizer(learning_rate).make_optimize_action(model)
    actions = [adam]
    actions = actions if callback is None else actions + callback

    Loop(actions, stop=iterations)()
    model.anchor(model.enquire_session())

