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
    def __init__(self, max_size=5000):
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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 504)
        self.layer_2 = nn.Linear(504, 126)
        self.layer_3 = nn.Linear(126, 126)
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Linear(126, 504),
            nn.ReLU(),
            nn.Linear(504, 126),
            nn.ReLU(),
            nn.Linear(126, action_dim),
            nn.Softmax(dim=1)
        ) for _ in range(27)])

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        ys = [layer(x) for layer in self.layers]
        return ys

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim_critic):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim_critic, 134)
        self.layer_2 = nn.Linear(134, 67)
        self.layer_3 = nn.Linear(67, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

class DDPG(object):
    def __init__(self, state_dim, action_dim, action_dim_critic):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim_critic).to(device)
        self.critic_target = Critic(state_dim, action_dim_critic).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.actor_l = []

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        xl = self.actor(state)
        lst = []
        n = env.rNum
        for i in xl:
            lst.append(i.cpu().data.numpy().flatten())
        zl = [lst[i:i + n] for i in range(0, len(lst), n)]
        return zl

    def train(self, replay_buffer, iterations, batch_size, discount, tau, policy_noise, noise_clip):
        for it in range(iterations):
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            ys = self.actor_target(next_state)
            next_action = torch.cat(ys, 1)
            next_action = next_action + torch.clamp(torch.tensor(np.random.normal(scale=policy_noise)), -noise_clip, noise_clip)
            next_action = torch.clamp(next_action, 0, 1)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + ((1 - done) * discount * target_Q).detach()
            current_Q = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            ys = self.actor(state)
            action = torch.cat(ys, 1)
            actor_loss = -self.critic(state, action).mean()
            self.actor_l.append(actor_loss.cpu().data.numpy())
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    div_avg = eval_episodes * 10
    for i in range(eval_episodes):
        obs = env.get_observation()[2]
        # print(obs)
        # xc
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            # print(action)
            # print(len(action))
            # xc
            obs, reward, done = env.step(action)
            avg_reward += reward
    avg_reward /= div_avg

    print("---------------------------------------")
    print("Average Reward over the Evaluation Step: %f" % (avg_reward))
    print("---------------------------------------")
    return avg_reward

if __name__ == "__main__":
    # Parameter settings
    seed = 0
    start_timesteps = 3000
    eval_freq = 100
    max_timesteps = 10000
    max_episode_steps = 10  # Define max_episode_steps here
    save_models = True
    expl_noise = 0.1
    batch_size = 32
    discount = 0.99
    tau = 0.005
    policy_noise = 0.05
    noise_clip = 0.5
    reward_threshold = 2.5  # Define reward_threshold here

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Environment()
    state_dim = env.get_observation()[2].shape[0]
    action_dim = env.rNum + env.gNum
    action_dim_critic = (action_dim) * env.rNum * env.gNum

    policy = DDPG(state_dim, action_dim, action_dim_critic)
    replay_buffer = ReplayBuffer()
    evaluations = [evaluate_policy(policy)]

    file_name = "%s_%s_%s" % ("DDPG", "MEC", str(seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    t0 = time.time()

    rew = []
    import timeit
    start = timeit.default_timer()
    dec_time_arr = []
    train_time_arr = []

    while total_timesteps < max_timesteps:
        if done:
            if total_timesteps != 0:
                print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
                train_start = timeit.default_timer()
                policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip)
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
            action = array
        else:
            dec_start = timeit.default_timer()
            action = policy.select_action(np.array(obs))
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
    file = f"./results/{dt_string}{s} sites_vNum{env.vv}_DDPG_rewards.txt"

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
        file_object.write("#### DDPG ########")
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
        file_object.write("rew = ")
        file_object.write(str(rew))
        file_object.write("\n")
    print("Training done in " + str(execution_time))
    print()
