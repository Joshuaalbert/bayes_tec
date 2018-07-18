from gpflow.actions import Action, Loop
from gpflow.training import AdamOptimizer
from ..logging import logging
import tensorflow as tf

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
        all_summaries = [tf.summary.scalar(p.pathname, tf.reshape(p.constrained_tensor, []))
                          for p in parameters if p.size == 1]
        all_summaries.append(tf.summary.scalar("optimisation/likelihood",
                                               self.model._likelihood_tensor))
        self._summary = tf.summary.merge(all_summaries)
        
    def run(self, ctx):
        if self.iteration % self.write_period == 0:
            summary = ctx.session.run(self._summary)
            self.writer.add_summary(summary,global_step=ctx.iteration)
        self.iteration += 1

def train_with_adam(model, learning_rate, iterations, callback=None):
    adam = AdamOptimizer(learning_rate).make_optimize_action(model)
    actions = [adam]
    actions = actions if callback is None else actions + [callback]

    Loop(actions, stop=iterations)()
    model.anchor(model.enquire_session())

