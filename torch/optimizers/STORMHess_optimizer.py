import torch
from torch.optim import Optimizer
from garage.torch.optimizers import SGDMOptimizer
from pdb import set_trace as bp
from dowel import logger, tabular


class STORMHessOptimizer(SGDMOptimizer):
    def __init__(self, params, gamma_0, eta_0=1, normalized=True,
                     const_stepsize=False, const_momentum=False,
                        hessian_batch_size=10000):

        super().__init__(params, gamma_0=gamma_0, eta_0=eta_0,
                            normalized=normalized,
                            const_stepsize=const_stepsize,
                            const_momentum=const_momentum,
                            moving_avg=True)
        self.hessian_batch_size = hessian_batch_size
        if self.normalized:
            self.alg_name = "NSTORM-Hess"
        else:
            self.alg_name = "STORM-Hess"


    def update_model_to_random_line_point(self):
        """
        update the parameter based on the displacement
        """
        b = torch.rand(1).item()
        for group in self.param_groups:
            with torch.no_grad():
                for p in group['params']:
                    state = self.state[p]
                    last_point, current_point = state['last_point'], state['current_point']
                    with torch.no_grad():
                        p.copy_(b * current_point + (1 - b) * last_point)
                        
    def save_current_point(self, ):

        for group in self.param_groups:
            with torch.no_grad():
                for p in group['params']:
                    state = self.state[p]
                    state['current_point'].copy_(p)
                    state['grad'] = p.grad.clone().detach()
                    #state['vector'] = state['current_point'] - state['last_point']
                    
    def get_vector_diff(self, group):
        vector = []
        with torch.no_grad():
            for p in group['params']:
                state = self.state[p]
                vector.append(state['current_point'] - state['last_point'])
        return vector
                    
                    
    def update_gamma(self):
        if not self.const_stepsize:
            if self.normalized:
                self.gamma = self.gamma_0 * 2 / (2 + self.iteration)
            else:
                self.gamma = self.gamma_0 * (2 / (2 + self.iteration)) ** 0.5

    def update_eta(self):
        if not self.const_momentum:
            self.eta = self.eta_0 * 2 / (2 + self.iteration)
            
    def compute_grad_estimator(self, p, hvp_full_p):
        state = self.state[p]
        grad = state['grad']
            
        if ('momentum_buffer' not in state) or (hvp_full_p is None) or (self.eta == 1):
            state['momentum_buffer'] = grad
        else:
            state['momentum_buffer'].add_(hvp_full_p)
            state['momentum_buffer'].mul_(1 - self.eta).add_(grad, alpha=self.eta)
        
        return state['momentum_buffer']
            
            
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.iteration += 1
        self.g_square_norm = 0
        self.grad_square_norm = 0

        self.save_current_point()
        self.update_gamma()
        self.update_eta()
        
        if self.iteration >= 1:
                # compute hessian vector
            self.update_model_to_random_line_point()
            with torch.enable_grad():
                new_loss, log_probs_loss = closure(batch_size=self.hessian_batch_size)
        
        for group in self.param_groups:
            i = 0
            if self.iteration >= 1:
                vector = self.get_vector_diff(group)
                jac = torch.autograd.grad(outputs=new_loss, inputs=group['params'], create_graph=True, retain_graph=True)
                hvp = torch.autograd.grad(outputs=jac, inputs=group['params'], grad_outputs=vector, )
                log_probs_grad = torch.autograd.grad(outputs=log_probs_loss, inputs=group['params'])
                log_probs_grad_scalar = sum([torch.matmul(torch.flatten(lp), torch.flatten(v)) for lp, v in zip(log_probs_grad, vector)])
            
            with torch.no_grad():
                for p in group['params']:
                    state = self.state[p]
                    if 'grad' not in state:
                        continue
                    self.grad_square_norm += state['grad'].norm(2).item() ** 2
                    if self.iteration >= 1:
                        hvp_full_p = hvp[i] + jac[i] * log_probs_grad_scalar
                        buf = self.compute_grad_estimator(p, hvp_full_p )
                    else:
                        buf = self.compute_grad_estimator(p, None )
                    self.g_square_norm += buf.norm(2).item() ** 2
                    i += 1

        if self.normalized:
            self.update_the_displacement(self.g_square_norm ** 0.5)
        else:
            self.update_the_displacement(1)

        self.log_step_stats(print_eta=True)        
            
        

    