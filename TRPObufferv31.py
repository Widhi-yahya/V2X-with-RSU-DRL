#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 21:12:01 2024

@author: widhi
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from MECenvirontment_V2X_V3 import Environment
import copy
import matplotlib.pyplot as plt
from datetime import datetime

class ReplayBuffer(object):
    def __init__(self, max_size=4000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor_layer_1 = nn.Linear(state_dim, 504)
        self.actor_layer_2 = nn.Linear(504, 126)
        self.actor_layer_3 = nn.Linear(126, 126)
        self.actor_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(126, 504),
            nn.ReLU(),
            nn.Linear(504, 126),
            nn.ReLU(),
            nn.Linear(126, action_dim),
            nn.Softmax(dim=1)
        ) for _ in range(27)])
        self.critic_layer_1 = nn.Linear(state_dim, 504)
        self.critic_layer_2 = nn.Linear(504, 126)
        self.critic_layer_3 = nn.Linear(126, 1)
    
    def forward(self, x):
        actor_x = F.relu(self.actor_layer_1(x))
        actor_x = F.relu(self.actor_layer_2(actor_x))
        actor_x = F.relu(self.actor_layer_3(actor_x))
        logits = [layer(actor_x) for layer in self.actor_layers]  # These are the logits before softmax
        
        # Clipping logits to prevent extreme values before softmax
        clipped_logits = [torch.clamp(logit, -5, 5) for logit in logits]
        
        # Applying softmax to convert logits to probabilities
        probabilities = [F.softmax(logit, dim=1) for logit in clipped_logits]
        
        critic_x = F.relu(self.critic_layer_1(x))
        critic_x = F.relu(self.critic_layer_2(critic_x))
        value = self.critic_layer_3(critic_x)
        
        return probabilities, value
    
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(self.size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.deepcopy(self.mu)

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


    
class TRPO:
    def __init__(self, state_dim, action_dim, action_dim_critic):
        self.actor_critic = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=1e-3, weight_decay=1e-4)
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.99
        self.tau = 0.95
        self.lam = 0.97
        self.delta = 0.01
        self.damping = 0.1
        self.cg_iters = 10
        self.actor_l = []
        self.noise = OUNoise(action_dim)  # Initialize the OUNoise instance
        self.reward_normalizer = RunningMeanStd()  # Initialize reward normalizer

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        ys, _ = self.actor_critic(state)
        lst = []
        n = env.rNum
        for i in ys:
            noise = torch.Tensor(self.noise.noise()).to(device)
            noisy_logits = i + noise
            probabilities = F.softmax(noisy_logits, dim=1)
            lst.append(probabilities.cpu().data.numpy().flatten())
        zl = [lst[i:i + n] for i in range(0, len(lst), n)]
        return zl

    def conjugate_gradient(self, Ax, b, nsteps, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)

        for i in range(nsteps):
            Avp_ = Ax(p)
            alpha = rdotr / torch.dot(p, Avp_)
            x += alpha * p
            r -= alpha * Avp_
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            print(f"Conjugate Gradient step {i + 1}, rdotr: {rdotr.item()}")

        return x

    def kl_divergence(self, states):
        with torch.no_grad():
            ys, _ = self.actor_critic(states)
            ys_old, _ = self.actor_critic(states)

        kl = 0
        for y, y_old in zip(ys, ys_old):
            kl += F.kl_div(y.log(), y_old, reduction='batchmean')

        return kl

    def Fvp(self, v, states):
        kl = self.kl_divergence(states)
        kl = kl.mean()
        grads = torch.autograd.grad(kl, self.actor_critic.parameters(), create_graph=True, allow_unused=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads if grad is not None])
        if flat_grad_kl.shape[0] != v.shape[0]:
            min_length = min(flat_grad_kl.shape[0], v.shape[0])
            flat_grad_kl = flat_grad_kl[:min_length]
            v = v[:min_length]
        print(f"flat_grad_kl shape: {flat_grad_kl.shape}, v shape: {v.shape}")
        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, self.actor_critic.parameters(), allow_unused=True)
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads if grad is not None]).data
        if flat_grad_grad_kl.shape[0] != v.shape[0]:
            min_length = min(flat_grad_grad_kl.shape[0], v.shape[0])
            flat_grad_grad_kl = flat_grad_grad_kl[:min_length]
            v = v[:min_length]
        return flat_grad_grad_kl + v * self.damping

    def train(self, replay_buffer, batch_size):
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
        states = torch.Tensor(batch_states).to(device)
        next_states = torch.Tensor(batch_next_states).to(device)
        rewards = torch.Tensor(batch_rewards).to(device)
        dones = torch.Tensor(batch_dones).to(device)

        actions = torch.Tensor(batch_actions).to(device)

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        _, values = self.actor_critic(states)
        _, next_values = self.actor_critic(next_states)

        Qvals = torch.zeros(len(rewards)).to(device)
        Qval = 0
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + self.gamma * Qval * (1 - dones[t])
            Qvals[t] = Qval

        advantage = Qvals - values.squeeze()
        advantage = advantage.unsqueeze(1)

        actor_loss = 0
        critic_loss = advantage.pow(2).mean()

        actor_outputs, _ = self.actor_critic(states)
        for i, actor_output in enumerate(actor_outputs):
            log_probs = torch.log(actor_output + 1e-8)
            actions_for_layer = actions[:, i].unsqueeze(-1) if actions.dim() == 3 else actions
            selected_action_log_probs = log_probs.gather(1, actions_for_layer.long())
            actor_loss += -(selected_action_log_probs * advantage).mean()

        actor_loss /= len(actor_outputs)

        loss = actor_loss + critic_loss
        self.actor_l.append(loss.cpu().data.numpy())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename, directory):
        torch.save(self.actor_critic.state_dict(), '%s/%s_actor_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor_critic.load_state_dict(torch.load('%s/%s_actor_critic.pth' % (directory, filename)))


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self._update_mean_var_count(self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    @staticmethod
    def _update_mean_var_count(mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = m2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    div_avg = eval_episodes * 10
    for i in range(eval_episodes):
        obs = env.get_observation()[2]
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done = env.step(action)
            avg_reward += reward
    avg_reward /= div_avg

    print("---------------------------------------")
    print("Average Reward over the Evaluation Step: %f" % (avg_reward))
    print("---------------------------------------")
    return avg_reward

# Set parameters
# Set parameters
env = Environment()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


seed = 0
start_timesteps = 2000
eval_freq = 100
max_timesteps = 10000
save_models = True
batch_size = 32
discount = 0.99
tau = 0.005
expl_noise = 0.001

file_name = "%s_%s_%s" % ("TRPO", "MEC", str(seed))
print("---------------------------------------")
print("Settings: %s" % (file_name))
print("---------------------------------------")

if not os.path.exists("./results"):
    os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

state_dim = env.get_observation()[2].shape[0]
action_dim = env.rNum + env.gNum
action_dim_critic = (action_dim) * env.rNum * env.gNum

policy = TRPO(state_dim, action_dim, action_dim_critic)
replay_buffer = ReplayBuffer()
evaluations = [evaluate_policy(policy)]

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')
max_episode_steps = 10

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
t0 = time.time()

import timeit
start = timeit.default_timer()

dec_time_arr = []
train_time_arr = []

reward_threshold = 2.5
reward_max = 3.8

while total_timesteps < max_timesteps:
    lst_reward = []
    if done:
        if total_timesteps != 0:
            print("++++++++++++++++++++++ Training Process ++++++++++++++++++++++++")
            print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
            train_start = timeit.default_timer()
            policy.train(replay_buffer, batch_size)
            train_end = timeit.default_timer()
            train_time_arr.append(train_end - train_start)

        if timesteps_since_eval >= eval_freq:
            timesteps_since_eval %= eval_freq
            reward_now = evaluate_policy(policy)
            evaluations.append(reward_now)
            print("evaluations =", evaluations)
            policy.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)
            if reward_now >= reward_threshold:
                policy.save(file_name, directory="./pytorch_models_max")
                np.save("./results/%s" % (file_name), evaluations)
                reward_threshold = reward_now

        obs = env.reset()
        obs1, obs2, obs = env.get_observation()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    if total_timesteps < start_timesteps:
        array, action = env.get_actions()
        print("--------Random Action-----------------")
        action = array
    else:
        dec_start = timeit.default_timer()
        action = policy.select_action(np.array(obs))
        print("--------NN Action-----------------")
        dec_end = timeit.default_timer()
        dec_time_arr.append(dec_end - dec_start)
        if expl_noise != 0:
            action = (action + np.random.normal(0, expl_noise, size=env.gNum + env.rNum)).clip(0, 1)

    actionsave = copy.deepcopy(action)
    new_obs, reward, done = env.step(action)
    done_bool = 0 if episode_timesteps + 1 == max_episode_steps else float(done)
    episode_reward += reward
    action = actionsave

    zx = []
    for i in [action]:
        zx.append(np.concatenate(i).ravel())
    action = np.array(zx).flatten()
    replay_buffer.add((obs, new_obs, action, reward, done_bool))

    obs = new_obs
    episode_timesteps += 1
    total_timesteps += 1
    timesteps_since_eval += 1

evaluations.append(evaluate_policy(policy))
policy.save("%s" % (file_name), directory="./pytorch_models")
np.save("./results/%s" % (file_name), evaluations)
stop = timeit.default_timer()
execution_time = stop - start
ev_reward = evaluate_policy(policy)

s = env.gNum * (env.rNum + 1)
from datetime import datetime
# Original print statements are now combined with file writing
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
file = f"./results/{dt_string}{s} sites_vNum{env.vv}_TRPO_rewards.txt"

with open(file, 'w') as f:
    f.write("="*40 + "{} sites vNum = {}".format(s, env.vv) + "="*40 + "\n")
    f.write("# of times made by NN: {}, average decision time: {} sec\n".format(len(dec_time_arr), np.mean(dec_time_arr)))
    f.write("Total numbers of reward: {}\n".format(len(evaluations)))
    
    if len(evaluations) <= 100:
        f.write("All reward record:\n{}\n".format(evaluations))
    
    conv_index = np.argmax(evaluations[10:]) + 10
    min_reward = np.argmin(evaluations[10:]) + 10
    conv_time = np.sum(train_time_arr[:100 * (conv_index + 1)])
    
    f.write("Convergence step: {}, value: {}\n".format(conv_index, evaluations[conv_index]))
    f.write("Convergence time: {} sec\n".format(conv_time))
    f.write("Minimum reward step: {}, value: {}\n".format(min_reward, evaluations[min_reward]))
    f.write("Average training time: {} sec, total: {}\n".format(np.mean(train_time_arr), len(train_time_arr)))
    f.writelines(str(r) + '\n' for r in evaluations)

    # If you also want to keep the print statements for console output
    print("="*40 + "{} sites vNum = {}".format(s, env.vv) + "="*40)
    print("# of times made by NN: {}, average decision time: {} sec".format(len(dec_time_arr), np.mean(dec_time_arr)))
    print("Total numbers of reward: {}".format(len(evaluations)))
    if len(evaluations) <= 100:
        print("All reward record:\n{}".format(evaluations))
    print("Convergence step: {}, value: {}".format(conv_index, evaluations[conv_index]))
    print("Convergence time: {} sec".format(conv_time))
    print("Minimum reward step: {}, value: {}".format(min_reward, evaluations[min_reward]))
    print("Average training time: {} sec, total: {}".format(np.mean(train_time_arr), len(train_time_arr)))

plt.figure(1)
plt.title("Reward")
plt.plot(evaluations, marker="x")
plt.show()

plt.figure(3)
plt.title("Actor Loss")
plt.plot(policy.actor_l, marker="x")
plt.show()

with open("train-new.txt", "a+") as file_object:
    file_object.seek(0)
    data = file_object.read(100)
    if len(data) > 0:
        file_object.write("\n")
    file_object.write("\n")
    file_object.write("############")
    file_object.write(dt_string)
    file_object.write("#############")
    file_object.write("\n")
    file_object.write("Evaluation reward = ")
    file_object.write(str(ev_reward))
    file_object.write("\n")
    file_object.write("Program Executed in ")
    file_object.write(str(execution_time))
    file_object.write("\n")
    file_object.write("AN Num = ")
    file_object.write(str(env.gNum))
    file_object.write("\n")
    file_object.write("Traffic = ")
    file_object.write(str(env.lam))
    file_object.write("\n")
    file_object.write("\n")

print("Training done in " + str(execution_time))
print()
