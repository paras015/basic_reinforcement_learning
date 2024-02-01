import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch import optim
from torch.distributions import Categorical
from gym.wrappers.record_video import RecordVideo

class PolicyNetwork:
    def __init__(self, num_observations, num_actions):
        self.num_observations = num_observations
        self.num_actions = num_actions

        self.network = nn.Sequential(
            nn.Linear(self.num_observations, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_actions),
            nn.Softmax(dim=-1)
        )
    
    def predict(self, observations):
        return self.network(torch.FloatTensor(observations))

if __name__ == "__main__":
    env_name = "CartPole-v1"
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder="./video_vpg",  episode_trigger=lambda t: t % 500 == 0)

    estimator = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)

    num_episodes = 1505
    batch_size = 10
    discount_factor = 0.99

    total_rewards, batch_rewards, batch_observations, batch_actions = [], [], [], []
    batch_counter = 1

    optimizer = optim.Adam(estimator.network.parameters(), lr = 0.01)
    action_space = np.arange(env.action_space.n)
    for current_episode in range(num_episodes):
        observation = env.reset()[0]
        rewards, actions, observations = [], [], []
        while True:
            action_probs = estimator.predict(observation).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)

            observations.append(observation)
            return_val = env.step(action)
            observation = return_val[0]
            reward = return_val[1]
            done = return_val[2]
            info = return_val[3]
            rewards.append(reward)
            actions.append(action)

            if done:
                r = np.full(len(rewards), discount_factor) ** np.arange(len(rewards)) * np.array(rewards)
                r = r[::-1].cumsum()[::-1]
                discounted_rewards = r - r.mean()
                total_rewards.append(sum(rewards))
                batch_rewards.extend(discounted_rewards)
                batch_observations.extend(observations)
                batch_actions.extend(actions)
                batch_counter += 1

                if batch_counter >= batch_size:
                    optimizer.zero_grad()
                    batch_rewards = torch.FloatTensor(batch_rewards)
                    batch_observationss = torch.FloatTensor(batch_observations)
                    batch_actions = torch.LongTensor(batch_actions)

                    logprob = torch.log(estimator.predict(batch_observations))
                    batch_actions = batch_actions.reshape(len(batch_actions), 1)
                    selected_logprobs = batch_rewards * torch.gather(logprob, 1, batch_actions).squeeze()
                    loss = -selected_logprobs.mean()

                    loss.backward()
                    optimizer.step()

                    batch_rewards, batch_observations, batch_actions = [], [], []
                    batch_counter = 1
                
                average_reward = np.mean(total_rewards[-100:])
                if current_episode % 100 == 0:
                    print(f"average of last 100 rewards as of episode {current_episode}: {average_reward:.2f}")
                break

    moving_average_num = 100
    def moving_average(x, n = moving_average_num):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[n:] - cumsum[:-n]) / float(n)

    # plotting
    plt.scatter(np.arange(len(total_rewards)), total_rewards, label='individual episodes', alpha=0.2)
    plt.plot(moving_average(total_rewards), label=f'moving average of last {moving_average_num} episodes')
    plt.title(f'Vanilla Policy Gradient on {env_name}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()