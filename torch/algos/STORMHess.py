"""Vanilla Policy Gradient integrated with STORMHess algorithm."""
import collections
import copy

import numpy as np
import torch.nn.functional as F
from dowel import tabular
from garage import log_performance
from garage.np import discount_cumsum
from garage.np import pad_batch_array
from garage.np.algos import RLAlgorithm
from garage.torch import compute_advantages, filter_valids
from garage.torch._functions import np_to_torch, zero_optim_grads
from garage.torch.optimizers import OptimizerWrapper
from garage.torch.algos import SGDM

import torch


class STORMHess(SGDM):
    """Vanilla Policy Gradient integrated with second-order information.


    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.Policy): Policy.
        value_function (garage.torch.value_functions.ValueFunction): The value
            function.
        sampler (garage.sampler.Sampler): Sampler.
        policy_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer
            for policy.
        vf_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer for
            value function.
        num_train_per_epoch (int): Number of train_once calls per epoch.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.

    """

    def __init__(
            self,
            env_spec,
            policy,
            value_function,
            sampler,
            policy_optimizer=None,
            vf_optimizer=None,
            num_train_per_epoch=1,
            discount=0.99,
            gae_lambda=1,
            center_adv=True,
            positive_adv=False,
            policy_ent_coeff=0.0,
            use_softplus_entropy=False,
            stop_entropy_gradient=False,
            entropy_method='no_entropy',
            neural_baseline=True
    ):
        
        super().__init__(env_spec,
            policy,
            value_function,
            sampler,
            policy_optimizer,
            vf_optimizer,
            num_train_per_epoch,
            discount,
            gae_lambda,
            center_adv,
            positive_adv,
            policy_ent_coeff,
            use_softplus_entropy,
            stop_entropy_gradient,
            entropy_method,
            neural_baseline)
        

    def _train_policy(self, obs, actions, rewards, advantages):
        r"""Train the policy.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, )`.
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated mean scalar value of policy loss (float).

        """

        # pylint: disable=protected-access
        zero_optim_grads(self._policy_optimizer._optimizer)
        loss = self._compute_loss_with_adv(obs, actions, rewards, advantages)

        def closure(batch_size=None):
            """
            This function takes a new sample for the hessian vector product and calls backward afterward.
            New Loss :return
            """
            zero_optim_grads(self._policy_optimizer._optimizer)
            if batch_size is None:
                eps = self.trainer.obtain_episodes(self.trainer.step_itr)
            else:
                eps = self.trainer.obtain_episodes(self.trainer.step_itr, batch_size=batch_size)
            obs = np_to_torch(eps.padded_observations)
            rewards = np_to_torch(eps.padded_rewards)
            valids = eps.lengths

            # if the value function is neural network, take the baselines by forward pass
            if self.neural_baseline:
                with torch.no_grad():
                    baselines = self._value_function(obs)
            else:
                temp_obs = [
                    self._value_function.predict({'observations': sub_obs})
                    for sub_obs in eps.observations_list
                ]
                baselines = pad_batch_array(np.concatenate(temp_obs), eps.lengths,
                                            self.max_episode_length)

                baselines = torch.from_numpy(baselines)

            obs_flat = np_to_torch(eps.observations)
            actions_flat = np_to_torch(eps.actions)
            rewards_flat = np_to_torch(eps.rewards)
            advs_flat = self._compute_advantage(rewards, valids, baselines)
            new_loss = self._compute_loss_with_adv(obs_flat, actions_flat, rewards_flat, advs_flat)
            
            log_probs_loss = -self._compute_log_probs_objective(obs_flat, actions_flat, rewards_flat).mean()
            #new_loss.backward(create_graph=True, retain_graph=True)
            return new_loss, log_probs_loss

        loss.backward(create_graph=True, retain_graph=True)
        self._policy_optimizer.step(closure=closure)

        return loss


    def _compute_log_probs_objective(self, obs, actions, rewards):
        r"""Compute log_probs objective.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N \dot [T], )`.

        Returns:
            torch.Tensor: Calculated objective values
                with shape :math:`(N \dot [T], )`.

        """
        del rewards
        log_likelihoods = self.policy(obs)[0].log_prob(actions)
        return log_likelihoods

