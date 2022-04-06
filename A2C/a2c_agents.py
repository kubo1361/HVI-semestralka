import numpy as np
import torch
import torch.nn.functional as F
from utils.agent import Agent
from tqdm.notebook import tqdm


class AgentA2C(Agent):
    def __init__(self, model_name, model, gamma=0.99, lr=0.001, beta_entropy=0.01, critic_loss_coef=0.5, id=0):
        super().__init__(model_name, model, gamma, lr, id) 
        
        self.actions_count = model.actions_count
        self.beta_entropy = beta_entropy
        self.critic_loss_coef = critic_loss_coef


    def train(self, workers, total_episodes, steps, write=True):
        # initial variables
        self.model.train()
        self.average_score = []
        self.average_steps = []
        best_avg = 0
        len_workers = len(workers)
        observations = []

        # initial observations
        for worker in workers:
            observations.append(torch.from_numpy(worker.reset()).float())
        observations = torch.stack(observations).to(self.device)

        tqdm_bar = tqdm(range(self.episode, total_episodes), desc="Episodes", total=total_episodes)

        while(self.episode < total_episodes):
            # iteration specific variables
            iter_critic_values = torch.zeros([steps, len_workers, 1])
            iter_actor_log_probs = torch.zeros([steps, len_workers, 1])
            iter_entropies = torch.zeros([steps, len_workers, 1])
            iter_rewards = torch.zeros([steps, len_workers, 1])
            iter_not_terminated = torch.ones([steps, len_workers, 1])

            for step in range(steps):
                # forward pass - actions and first critic values
                step_actor, step_critic = self.model(observations)
                step_actor, step_critic = step_actor.cpu(), step_critic.cpu()

                # step specific variables
                step_rewards = torch.zeros([len_workers, 1])
                step_not_terminated = torch.ones(
                    [len_workers, 1], dtype=torch.int8)
                observations = []

                # extract actions
                step_probs = F.softmax(step_actor, dim=-1)
                if workers[0].isDiscrete():
                    step_actions = step_probs.multinomial(num_samples=1).detach()
                else:
                    step_actions = step_actor.detach()


                # step entropy calculation
                step_log_probs = F.log_softmax(step_actor, dim=-1)
                step_entropies = (
                    step_log_probs * step_probs).sum(1, keepdim=True)

                if workers[0].isDiscrete():
                    step_log_probs_policy = step_log_probs.gather(1, step_actions)
                else:
                    step_log_probs_policy = step_log_probs
                
                # update iteration steps
                iter_critic_values[step] = step_critic
                iter_entropies[step] = step_entropies
                iter_actor_log_probs[step] = step_log_probs_policy

                for worker in range(len_workers):
                    # Apply actions to workers enviroments
                    worker_observation, step_rewards[worker, 0], worker_terminated = workers[worker].step(
                        step_actions[worker].item())

                    # reset terminated workers
                    if worker_terminated:
                        step_not_terminated[worker, 0] = 0
                        worker_observation = workers[worker].reset()
                        tqdm_bar.update(1)

                    # append new observations
                    observations.append(torch.from_numpy(
                        worker_observation).float())

                # update observations, rewards and terminated workers
                observations = torch.stack(observations).to(self.device)
                iter_rewards[step] = step_rewards
                iter_not_terminated[step] = step_not_terminated

            # forward pass - critic values after performing action
            with torch.no_grad():
                _, critic_values = self.model(observations)
                critic_values = critic_values.detach().cpu()

            # compute advantage - we compute steps backwards
            # with their respective critic values for each step
            advantages = torch.zeros([steps, len_workers, 1])
            for step in reversed(range(steps)):
                critic_values = iter_rewards[step] + \
                    (self.gamma * critic_values * iter_not_terminated[step])

                advantages[step] = critic_values - iter_critic_values[step]

            # standard score normalization of advantage
            advantages = (advantages - torch.mean(advantages)) / \
                (torch.std(advantages) + 1e-5)

            advantages_detached = advantages.detach()

            # average reward for statistics
            average_reward = iter_rewards.mean().detach()

            # calculate losses
            critic_loss = (advantages**2).mean() * self.critic_loss_coef
            actor_loss = - (iter_actor_log_probs * advantages_detached).mean()

            entropy_loss = (iter_entropies.mean() * self.beta_entropy)

            # clear gradients
            self.optimizer.zero_grad()

            # calculate final loss
            loss = actor_loss + critic_loss + entropy_loss

            # backward pass with our total loss https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/
            loss.backward()

            # gradient clipping for exploding gradients https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

            # optimizer step
            self.optimizer.step()

            # stats
            if self.episode % 10 == 0 and self.episode > 0:

                # average for last 10 scores
                avg_score = np.average(self.average_score[-100:])
                avg_steps = np.average(self.average_steps[-100:])
                # save model on new best average score
                if avg_score > best_avg:
                    best_avg = avg_score
                    print('Saving model, best score is: ', best_avg)
                    self.save_model()

                tqdm_bar.set_postfix({'Average score': avg_score, "Average steps": avg_steps, "Total loss": loss.item()})

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

            if self.episode % 500 == 0 and self.episode > 0:
                self.average_score = self.average_score[-100:]
                self.average_steps = self.average_steps[-100:]
