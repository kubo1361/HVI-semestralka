import numpy as np
import torch
import torch.nn.functional as F
from utils.agent import Agent
from tqdm.notebook import tqdm
from A2C.workers import Worker


class DiscreteA2C(Agent):
    def __init__(self, model_name, model, gamma=0.99, lr=0.0001, beta_entropy=0.01, critic_loss_coef=0.5, id=0):
        super().__init__(model_name, model, gamma, lr, id)

        self.actions_count = model.actions_count
        self.beta_entropy = beta_entropy
        self.critic_loss_coef = critic_loss_coef

    def _init_workers(self, envs):
        workers = []
        for id_w in range(len(envs)):
            w = Worker(id_w, envs[id_w], self)
            workers.append(w)
        return workers

    def _reset_iter_variables(self, steps, workers):
        len_workers = len(workers)

        crit_vals = torch.zeros(
            [steps, len_workers, 1]).type(self.tensor).to(self.device)
        actor_log_probs = torch.zeros(
            [steps, len_workers, 1]).type(self.tensor).to(self.device)
        entropies = torch.zeros([steps, len_workers, 1]).type(
            self.tensor).to(self.device)
        rewards = torch.zeros([steps, len_workers, 1]).type(
            self.tensor).to(self.device)
        not_terminated = torch.ones(
            [steps, len_workers, 1]).type(self.tensor).to(self.device)
        return crit_vals, actor_log_probs, entropies, rewards, not_terminated

    def _env_init_obs(self, workers):
        observations = []
        for worker in workers:
            observations.append(torch.from_numpy(
                worker.reset()).type(self.tensor))
        observations = torch.stack(observations).to(self.device)
        return observations

    def _extract_action(self, x):
        return x.item()

    def _env_step(self, workers, actions, tqdm_bar):
        len_workers = len(workers)
        step_observations = []
        step_rewards = torch.zeros([len_workers, 1])
        step_not_terminated = torch.ones(
            [len_workers, 1], dtype=torch.int8)

        for id_w in range(len_workers):
            # Apply actions to workers enviroments
            worker_observation, step_rewards[id_w, 0], worker_terminated = workers[id_w].step(
                self._extract_action(actions[id_w]))

            # reset terminated workers
            if worker_terminated:
                step_not_terminated[id_w, 0] = 0
                worker_observation = workers[id_w].reset()
                tqdm_bar.update(1)

            # append new observations
            step_observations.append(torch.from_numpy(
                worker_observation).type(self.tensor))

        # update observations, rewards and terminated workers
        step_observations = torch.stack(step_observations).to(self.device)
        return step_observations, step_rewards, step_not_terminated

    def _agent_step(self, observations):
        # forward pass - actions and first critic values
        step_actor, step_critic = self.model(observations)

        # extract actions
        step_probs = F.softmax(step_actor, dim=-1)
        step_actions = step_probs.multinomial(num_samples=1).detach()

        # step entropy calculation
        step_log_probs = F.log_softmax(step_actor, dim=-1)
        step_entropies = (
            step_log_probs * step_probs).sum(1, keepdim=True)

        step_log_probs_policy = step_log_probs.gather(1, step_actions)

        return step_actions, step_log_probs_policy, step_critic, step_entropies

    def _getCriticValues(self, observations):
        with torch.no_grad():
            _, critic_values = self.model(observations)
        return critic_values.detach()

    def _compute_advantage_iter(self, observations, workers, iter_rewards, iter_not_terminated, iter_critic_values):
        steps = iter_rewards.size(dim=0)
        # forward pass - critic values after performing action
        critic_values = self._getCriticValues(observations)

        # compute advantage - we compute steps backwards
        # with their respective critic values for each step
        advantages = torch.zeros([steps, len(workers), 1])
        for step in reversed(range(steps)):
            critic_values = iter_rewards[step] + \
                (self.gamma * critic_values * iter_not_terminated[step])

            advantages[step] = critic_values - iter_critic_values[step]

        # standard score normalization of advantage
        advantages = (advantages - torch.mean(advantages)) / \
            (torch.std(advantages) + 1e-5)

        return advantages

    def _compute_loss_iter(self, advantages, iter_actor_log_probs, iter_entropies):
        # calculate losses
        advantages_detached = (advantages.detach()).to(self.device)
        critic_loss = (advantages**2).mean() * self.critic_loss_coef
        actor_loss = - (iter_actor_log_probs * advantages_detached).mean()
        entropy_loss = (iter_entropies.mean() * self.beta_entropy)

        # calculate final loss
        return actor_loss, critic_loss, entropy_loss

    def train(self, envs, total_episodes, steps, write=True):
        # initial variables
        self.model.train()
        self.average_score = []
        self.average_steps = []
        avg_score = 0
        avg_steps = 0
        iter = 0
        workers = self._init_workers(envs)
        len_workers = len(workers)
        best_avg = 0
        tqdm_bar = tqdm(range(self.episode, total_episodes),
                        desc="Episodes", total=total_episodes)

        observations = self._env_init_obs(workers)
        while(self.episode < total_episodes):
            # iteration specific variables
            iter_critic_values, iter_actor_log_probs, iter_entropies, iter_rewards, iter_not_terminated = self._reset_iter_variables(
                steps, workers)

            for step in range(steps):
                step_actions, iter_actor_log_probs[step], iter_critic_values[step], iter_entropies[step] = self._agent_step(
                    observations)
                observations, iter_rewards[step], iter_not_terminated[step] = self._env_step(
                    workers, step_actions, tqdm_bar)

            # average reward for statistics
            average_reward = iter_rewards.mean().detach()

            # Advantage
            advantages = self._compute_advantage_iter(
                observations, workers, iter_rewards, iter_not_terminated, iter_critic_values)

            # loss
            actor_loss, critic_loss, entropy_loss = self._compute_loss_iter(
                advantages, iter_actor_log_probs, iter_entropies)
            loss = actor_loss + critic_loss + entropy_loss

            # clear gradients
            self.optimizer.zero_grad()

            # backward pass with our total loss https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/
            loss.backward()

            # gradient clipping for exploding gradients https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

            # optimizer step
            self.optimizer.step()

            # stats
            tqdm_bar.set_postfix(
                {"Iteration": iter, 'Average score': avg_score, "Average steps": avg_steps, "Total loss": loss.item()})

            if iter % 5 == 0 and iter > 0:

                # average for last 10 scores
                avg_score = np.average(self.average_score[-100:])
                avg_steps = np.average(self.average_steps[-100:])

                # write to file - log
                if write:
                    self._write_stats([
                        ("Average steps", avg_steps),
                        ("Average score", avg_score),
                        ("Average reward", average_reward),
                        ("Actor Loss", actor_loss.item()),
                        ("Critic Loss", critic_loss.item()),
                        ("Entropy Loss", entropy_loss.item()),
                        ("Total Loss", loss.item())
                    ])

            if iter % 50 == 0 and iter > 0:
                # save model on new best average score
                if avg_score > best_avg:
                    best_avg = avg_score
                    print('Saving model, best score is: ', best_avg)
                    self.save_model()

            if iter % 100 == 0 and iter > 0:
                self.average_score = self.average_score[-100:]
                self.average_steps = self.average_steps[-100:]

            iter += 1

    def act(self, observation):
        self.model.eval()
        obs = torch.from_numpy(observation).float()
        actor, _ = self.model(obs.to(self.device))
        step_probs = F.softmax(actor, dim=-1)
        step_actions = step_probs.multinomial(num_samples=1)
        return self._extract_action(step_actions)
