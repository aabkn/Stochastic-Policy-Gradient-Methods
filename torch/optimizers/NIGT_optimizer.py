import torch
from torch.optim import Optimizer
from garage.torch.optimizers import SGDMOptimizer
from pdb import set_trace as bp
from dowel import logger, tabular


class NIGTOptimizer(SGDMOptimizer):
    def __init__(self, params, gamma_0, eta_0=1, normalized=True,
                    const_stepsize=False, const_momentum=False):

        super().__init__(params, gamma_0=gamma_0, eta_0=eta_0,
                            normalized=normalized,
                            const_stepsize=const_stepsize,
                            const_momentum=const_momentum,
                            moving_avg=True)

        if self.normalized:
            self.alg_name = "NIGT"
        else:
            self.alg_name = "IGT"


    def update_model_to_line_point(self):
        """
        update the parameter based on the displacement
        """
        self.save_current_point()
        if self.eta is None:
            b = 1
        else:
            b = 1 / self.eta
        
        for group in self.param_groups:
            with torch.no_grad():
                for p in group['params']:
                    state = self.state[p]
                    last_point, current_point = state['last_point'], state['current_point']
                    with torch.no_grad():
                        p.copy_(b * current_point + (1 - b) * last_point)

    def update_gamma(self):
        if not self.const_stepsize:
            self.gamma = self.gamma_0 * (2 / (self.iteration + 2)) 

    def update_eta(self):
        if not self.const_momentum:
            self.eta = self.eta_0 * (2 / (2 + self.iteration)) ** (4. / 5)
            
    def step(self):
        """Performs a single optimization step.
        """

        self.iteration += 1
        self.g_square_norm = 0
        self.grad_square_norm = 0

        self.update_gamma()
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
            
    