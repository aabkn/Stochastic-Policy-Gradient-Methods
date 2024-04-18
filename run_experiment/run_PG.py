#!/usr/bin/env python3
import argparse
import os
from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.experiment import deterministic
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler, RaySampler, MultiprocessingSampler
from garage.torch.algos import STORMHess, SGDM, NIGT
from garage.torch.optimizers import OptimizerWrapper
from garage.torch.optimizers.SGDM_optimizer import SGDMOptimizer
from garage.torch.optimizers.STORMHess_optimizer import STORMHessOptimizer
from garage.torch.optimizers.NIGT_optimizer import NIGTOptimizer
from garage.torch.policies import GaussianMLPPolicy, SoftmaxMLPPolicy, CategoricalMLPPolicy
from garage.trainer import Trainer

from garage.torch import set_gpu_mode

import torch

parser = argparse.ArgumentParser(description='Training an RL Agent on Different Environments.')
parser.add_argument('-method', type=str, default="sgd",
                    help='optimization method')
parser.add_argument('-env', type=str, default="acrobot",
                    help='environment')
parser.add_argument('-seed', type=int, default=11,
                    help='random seed')
parser.add_argument('-torch_seed', type=int, default=11,
                    help='init random seed')
parser.add_argument('-epochs', type=int, default=1000,
                    help='number of epochs')
parser.add_argument('-batch_size', type=int, default=1000,
                    help='batch size')
parser.add_argument('-hessian_batch_size', type=int, default=32,
                    help='hessian batch size (for STORMhess)')
parser.add_argument('-gamma_0', type=float, default=0.1,
                    help='initial step-size')
parser.add_argument('-eta_0', type=float, default=1,
                    help='initial momentum parameter')
parser.add_argument('-logdir', type=str, default="path_to_dir",
                    help='path to directory for logs')
parser.add_argument('--debug', action='store_true', default=False,
                    help='save logs to the debug directory')

args = parser.parse_args()

if args.method not in ['nsgdm', 'sgd', 'stormhess', 'nstormhess', 'nigt']:
    raise Exception("The methods {} is not available.".format(args.method))

if args.env not in ['walker', 'acrobot', 'cartpole', 'halfcheetah', 'hopper', 'humanoid', 'reacher', 'swimmer']:
    raise Exception("The environment {} is not available.".format(args.env))



base_logdir = args.logdir

if args.debug:
    base_logdir = "path_to_debug_dir"

if args.method in ['stormhess', 'nstormhess']:
    logdir = os.path.join(base_logdir,
                        '{}/{}/logs_bsize={}_hbsize={}_g0={}_eta0={}_seed={}'.format(args.env, args.method, args.batch_size, args.hessian_batch_size, args.gamma_0, args.eta_0, args.seed))
else:
    logdir = os.path.join(base_logdir,
                        '{}/{}/logs_bsize={}_g0={}_seed={}'.format(args.env, args.method, args.batch_size, args.gamma_0, args.seed))




@wrap_experiment(log_dir=logdir, archive_launch_repo=False, x_axis='Evaluation/Iteration')
def run_PG(ctxt=None, seed=3, torch_seed=11):
    set_seed(seed, torch_seed)


    runner = Trainer(ctxt)

    n_epochs = args.epochs
    sampler_batch_size = args.batch_size
    max_episode_length = None
    if args.env == 'acrobot':
        env = GymEnv('Acrobot-v1')
    elif args.env == 'cartpole':
        env = GymEnv('CartPole-v1')
        max_episode_length = 200
    elif args.env == 'halfcheetah':
        env = normalize(GymEnv('HalfCheetah-v2'))
        max_episode_length = 500
    elif args.env == 'hopper':
        env = GymEnv('Hopper-v2')
    elif args.env == 'humanoid':
        env = GymEnv('Humanoid-v2')
    elif args.env == 'reacher':
        env = (GymEnv('Reacher-v2'))
    elif args.env == 'swimmer':
        env = GymEnv('Swimmer-v2')
        max_episode_length = 500
    elif args.env == 'walker':
        env = GymEnv('Walker2d-v2')

    if max_episode_length is None:
        max_episode_length = env.spec.max_episode_length

    env.seed(seed)
    env.action_space.seed(seed)
    env.action_space.np_random.seed(seed)

    if args.env in ['acrobot', 'cartpole']:

        policy = SoftmaxMLPPolicy(env.spec, hidden_sizes=[32, 32],
                                 hidden_nonlinearity=torch.nn.Tanh(),
                                 output_nonlinearity=None,)

        value_function = LinearFeatureBaseline(env_spec=env.spec)
        sampler = LocalSampler(agents=policy,
                               envs=env,
                               max_episode_length=max_episode_length,
                               seed=seed )
    else:
        policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[64, 64], )

        value_function = LinearFeatureBaseline(env_spec=env.spec)

        sampler = MultiprocessingSampler(agents=policy,
                             envs=env,
                             max_episode_length=max_episode_length,
                             seed=seed)

    if args.method == 'nsgdm':
        policy_optimizer = OptimizerWrapper((SGDMOptimizer, {"eta_0": args.eta_0,
                                                            "gamma_0": args.gamma_0,
                                                            "normalized": True}), policy)
        algo = SGDM(env_spec=env.spec,
                     policy=policy,
                     value_function=value_function,
                     sampler=sampler,
                     discount=0.99,
                     center_adv=False,
                     policy_optimizer=policy_optimizer,
                     neural_baseline=False)
    elif args.method == 'sgd':
        policy_optimizer = OptimizerWrapper((SGDMOptimizer, {"gamma_0": args.gamma_0,
                                                            "moving_avg": False,
                                                            "normalized": False}), policy)
        algo = SGDM(env_spec=env.spec,
                     policy=policy,
                     value_function=value_function,
                     sampler=sampler,
                     discount=0.99,
                     center_adv=False,
                     policy_optimizer=policy_optimizer,
                     neural_baseline=False)
    elif args.method in ['stormhess', 'nstormhess']:
        normalized = False
        if args.method == 'nstormhess':
            normalized = True
        policy_optimizer = OptimizerWrapper((STORMHessOptimizer, {"eta_0": args.eta_0,
                                                            "gamma_0": args.gamma_0,
                                                            "normalized": normalized,
                                                            "hessian_batch_size": args.hessian_batch_size}), policy)
        algo = STORMHess(env_spec=env.spec,
                     policy=policy,
                     value_function=value_function,
                     sampler=sampler,
                     discount=0.99,
                     center_adv=False,
                     policy_optimizer=policy_optimizer,
                     neural_baseline=False)

    elif args.method == 'nigt':
        policy_optimizer = OptimizerWrapper((NIGTOptimizer, {"eta_0": args.eta_0,
                                                            "gamma_0": args.gamma_0,
                                                            "normalized": True}), policy)
        algo = NIGT(env_spec=env.spec,
                     policy=policy,
                     value_function=value_function,
                     sampler=sampler,
                     discount=0.99,
                     center_adv=False,
                     policy_optimizer=policy_optimizer,
                     neural_baseline=False)
    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)


    runner.setup(algo, env)
    runner.train(n_epochs=n_epochs, batch_size=sampler_batch_size)


run_PG(seed=args.seed, torch_seed=args.torch_seed)
