"""Vanilla Policy Gradient integrated with Implicit Gradient Transport."""
import collections
import copy

import numpy as np
import torch.nn.functional as F
from dowel import tabular
from garage import (log_performance, obtain_evaluation_episodes)
from garage.np import discount_cumsum
from garage.np import pad_batch_array
from garage.np.algos import RLAlgorithm
from garage.torch import compute_advantages, filter_valids
from garage.torch._functions import np_to_torch, zero_optim_grads
from garage.torch.optimizers import OptimizerWrapper
from garage.torch.algos import SGDM
import time
import torch


class NIGT(SGDM):
    """Vanilla Policy Gradient integrated with Implicit Gradient Transport.


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


    def _train_once(self, itr):
        """Train the algorithm once.

        Args:
            itr (int): Iteration number.

        Returns:
            numpy.float64: Calculated mean value of undiscounted returns.

        """
        self._policy_optimizer.update_model_to_line_point()
        eps = self.trainer.obtain_episodes(self.trainer.step_itr)
                
        obs = np_to_torch(eps.padded_observations)
        rewards = np_to_torch(eps.padded_rewards)
        returns = np_to_torch(
            np.stack([
                discount_cumsum(reward, self.discount)
                for reward in eps.padded_rewards
            ]))
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

        if self._maximum_entropy:
            policy_entropies = self._compute_policy_entropy(obs)
            rewards += self._policy_ent_coeff * policy_entropies

        obs_flat = np_to_torch(eps.observations)
        actions_flat = np_to_torch(eps.actions)
        rewards_flat = np_to_torch(eps.rewards)
        returns_unflat = filter_valids(returns, valids)
        returns_flat = torch.cat(returns_unflat)
        advs_flat = self._compute_advantage(rewards, valids, baselines)

        with torch.no_grad():
            policy_loss_before = self._compute_loss_with_adv(
                obs_flat, actions_flat, rewards_flat, advs_flat)
            kl_before = self._compute_kl_constraint(obs)

        self._train(obs_flat, actions_flat, rewards_flat, returns_flat,
                    advs_flat)

        # if the value function is not a neural network, do the fitting
        if not self.neural_baseline:
            self._fit_baseline_with_data(returns_unflat, eps, baselines)

        with tabular.prefix('Training/'):
                    tabular.record('NumEpisodes', len(eps.lengths))  
            
        with torch.no_grad():
            policy_loss_after = self._compute_loss_with_adv(
                obs_flat, actions_flat, rewards_flat, advs_flat)
            kl_after = self._compute_kl_constraint(obs)
            policy_entropy = self._compute_policy_entropy(obs)

        with tabular.prefix(self.policy.name):
            tabular.record('/LossBefore', policy_loss_before.item())
            tabular.record('/LossAfter', policy_loss_after.item())
            tabular.record('/dLoss',
                           (policy_loss_before - policy_loss_after).item())
            tabular.record('/KLBefore', kl_before.item())
            tabular.record('/KL', kl_after.item())
            tabular.record('/Entropy', policy_entropy.mean().item())

        self._old_policy.load_state_dict(self.policy.state_dict())

        #return np.mean(undiscounted_returns)

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        if self.trainer is None:
            self.trainer = trainer
            
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        last_return = None
        for _ in trainer.step_epochs():
            for _ in range(self._n_samples):

                eval_episodes = obtain_evaluation_episodes(
                    self.policy, self._eval_env,
                    num_eps=self._num_eval_episodes,
                    max_episode_length=self.max_episode_length)
                    
                last_return = log_performance(trainer.step_itr,
                                               eval_episodes,
                                               discount=self._discount)
                
                last_return = self._train_once(trainer.step_itr)
                
                trainer.step_itr += 1

        return last_return

    def _train(self, obs, actions, rewards, returns, advs):
        r"""Train the policy and value function with minibatch.

        Args:
            obs (torch.Tensor): Observation from the environment with shape
                :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment with shape
                :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards with shape :math:`(N, )`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.
            advs (torch.Tensor): Advantage value at each step with shape
                :math:`(N, )`.

        """
        for dataset in self._policy_optimizer.get_minibatch(
                obs, actions, rewards, advs):
            self._train_policy(*dataset)

        if self.neural_baseline:
            for dataset in self._vf_optimizer.get_minibatch(obs, returns):
                self._train_value_function(*dataset)

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

        loss.backward(create_graph=True, retain_graph=True)
        self._policy_optimizer.step()

        return loss