from gpflow.actions import Action, Loop
from gpflow.training import AdamOptimizer
from ..logging import logging

class PrintAction(Action):
    def __init__(self, model, text):
        self.model = model
        self.text = text
        
    def run(self, ctx):
        likelihood = ctx.session.run(self.model.likelihood_tensor)
        logging.warning('{}: iteration {} likelihood {:.4f}'.format(self.text, ctx.iteration, likelihood))

class SendSummary(Action):
    def __init__(self, model, writer):
        self.model = model
        self.writer = writer
        self.summary = tf.summary.merge_all()
        
    def run(self, ctx):
        summary = ctx.session.run(self.summary)
        self.writer.add_summary(summary,global_step=ctx.iteration)

def train_with_adam(model, learning_rate, iterations, callback=None):
    adam = AdamOptimizer(learning_rate).make_optimize_action(model)
    actions = [adam]
    actions = actions if callback is None else actions + [callback]

    Loop(actions, stop=iterations)()
    model.anchor(model.enquire_session())

