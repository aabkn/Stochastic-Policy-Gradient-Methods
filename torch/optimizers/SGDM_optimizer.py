import torch
from torch.optim import Optimizer
from pdb import set_trace as bp
from dowel import logger, tabular


class SGDMOptimizer(Optimizer):
    def __init__(self, params, gamma_0, eta_0=2,
                    normalized=True, moving_avg=False,
                    const_stepsize=False, const_momentum=False):


        self.gamma_0 = gamma_0
        self.eta_0 = eta_0
        self.gamma = self.gamma_0
        self.eta = None

        self.iteration = -1
        self.g_square_norm = 0
        self.grad_square_norm = 0

        self.normalized = normalized
        self.moving_avg = moving_avg
        self.const_stepsize = const_stepsize
        self.const_momentum = const_momentum

        defaults = dict()
        if self.normalized:
            if self.moving_avg:
                self.alg_name = "NSGDM"
            else:
                self.alg_name = "NSGD"
        elif self.moving_avg:
            self.alg_name = "SGDM"
        else:
            self.alg_name = "SGD"


        super(SGDMOptimizer, self).__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['current_point'] = torch.zeros_like(p)
                state['last_point'] = torch.zeros_like(p)

    def update_the_displacement(self, total_norm):
        """
        update the parameter based on the displacement
        """

        # clip the displacement
        for group in self.param_groups:
            with torch.no_grad():
                for p in group['params']:
                    state = self.state[p]
                    buf = state['momentum_buffer']
                    last_point, current_point = state['last_point'], state['current_point']

                    p.copy_(current_point - self.gamma / total_norm * buf)
                    last_point.copy_(current_point)

    def save_current_point(self, ):

        for group in self.param_groups:
            with torch.no_grad():
                for p in group['params']:
                    state = self.state[p]
                    state['current_point'].copy_(p)

    def update_gamma(self):
        if not self.const_stepsize:
            if self.moving_avg:
                self.gamma = 2 * self.gamma_0 / (2 + self.iteration)
            else:
                self.gamma = self.gamma_0 * (2 / (2 + self.iteration)) ** (2. /3)

    def update_eta(self):
        if not self.const_momentum:
            self.eta = self.eta_0 * (2 / (2 + self.iteration)) ** (2. / 3)

    def compute_grad_estimator(self, p):
        d_p = p.grad.clone()
        state = self.state[p]
        if self.moving_avg:
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.clone(d_p).detach()
            else:
                state['momentum_buffer'].mul_(1 - self.eta).add_(d_p, alpha=self.eta)
        else:
            state['momentum_buffer'] = torch.clone(d_p).detach()
        return state['momentum_buffer']


    def step(self):
        """Performs a single optimization step.
        """

        self.iteration += 1
        self.g_square_norm = 0
        self.grad_square_norm = 0

        self.save_current_point()
        self.update_gamma()
        if self.moving_avg:
            self.update_eta()

        for group in self.param_groups:
            with torch.no_grad():
                for p in group['params']:
                    if p.grad is None:
                        continue
                    self.grad_square_norm += p.grad.clone().norm(2).item() ** 2
                    buf = self.compute_grad_estimator(p)
                    self.g_square_norm += buf.norm(2).item() ** 2

        if self.normalized:
            self.update_the_displacement(self.g_square_norm ** 0.5)
        else:
            self.update_the_displacement(1)

        self.log_step_stats(print_eta=self.moving_avg)


    def log_step_stats(self, print_eta=False):
        with tabular.prefix(self.alg_name + '/'):
            tabular.record('norm of g', self.g_square_norm ** 0.5)
            tabular.record('norm of gradient', self.grad_square_norm ** (1. / 2))
            tabular.record('gamma', self.gamma)
            if print_eta:
                tabular.record('eta', self.eta)
