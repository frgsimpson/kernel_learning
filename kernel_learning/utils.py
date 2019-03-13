# Copyright (C) PROWLER.io 2018
#
# Licensed under the Apache License, Version 2.0

from gpflow.training import AdamOptimizer
from gpflow.actions import Loop, Action

PRINT_INTERVAL = 1000


def run_adam(model, lr, iterations, callback=None):
    adam = AdamOptimizer(lr).make_optimize_action(model)
    actions = [adam] if callback is None else [adam, callback]
    loop = Loop(actions, step=PRINT_INTERVAL, stop=iterations)()
    model.anchor(model.enquire_session())


class PrintAction(Action):
    def __init__(self, model, text):
        self.model = model
        self.text = text

    def run(self, ctx):
        likelihood = ctx.session.run(self.model.likelihood_tensor)
        print('{}: iteration {} likelihood {:.4f}'.format(self.text, ctx.iteration, likelihood))
