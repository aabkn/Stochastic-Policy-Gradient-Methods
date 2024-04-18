"""PyTorch optimizers."""
from garage.torch.optimizers.conjugate_gradient_optimizer import (
    ConjugateGradientOptimizer)
from garage.torch.optimizers.differentiable_sgd import DifferentiableSGD
from garage.torch.optimizers.differentiable_sgd import DifferentiableSGD
from garage.torch.optimizers.optimizer_wrapper import OptimizerWrapper
from garage.torch.optimizers.SGD_optimizer import SGDOptimizer
from garage.torch.optimizers.SGDM_optimizer import SGDMOptimizer
from garage.torch.optimizers.STORMHess_optimizer import STORMHessOptimizer
from garage.torch.optimizers.SHARP_optimizer import SHARPOptimizer
from garage.torch.optimizers.NIGT_optimizer import NIGTOptimizer
__all__ = [
    'OptimizerWrapper', 'ConjugateGradientOptimizer', 'DifferentiableSGD',
     'SGDOptimizer', 'STORMHessOptimizer', 'SHARPOptimizer', 'SGMDOptimizer', 'NIGTOptimizer'
]
