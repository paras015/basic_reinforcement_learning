import gym
import pandas
from gym.wrappers.record_video import RecordVideo
import numpy
import random
from functools import reduce

class Qlean:
    def __init__(self, actions, alpha, epsilon, gamma):
        self.actions = actions 
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.q = {}

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)

            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)
            action = self.actions[i]
        return action
    
    def learnQ(self, state, action, reward, value):
        oldV = self.q.get((state, action), None)

        if oldV is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldV + self.alpha * (value - oldV)

    def learn(self, state, action, reward, state1, action1):
        maxQ = self.getQ(state1, action1)
        self.learnQ(state, action, reward, reward + self.gamma * maxQ)

    


def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return numpy.digitize(x=[value], bins=bins)[0]

if __name__ == '__main__':
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env, video_folder="./video_sarsa",  episode_trigger=lambda t: t % 5000 == 0)
    n_bins = 8
    n_bins_angle = 10
    max_number_of_steps = 200
    last_time_steps = numpy.ndarray(0)

    cart_position_bins = pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
    pole_angle_bins = pandas.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]
    cart_velocity_bins = pandas.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    angle_rate_bins = pandas.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]

    qlearn = Qlean(range(env.action_space.n), alpha=0.1, epsilon=0.1, gamma=0.9)

    for i_episode in range(10001):
        observation = env.reset()
        cart_position = observation[0][0]
        pole_angle = observation[0][1]
        cart_velocity = observation[0][2]
        angle_rate_of_change = observation[0][3]

        state = build_state([to_bin(cart_position, cart_position_bins),
                            to_bin(pole_angle, pole_angle_bins),
                            to_bin(cart_velocity, cart_velocity_bins),
                            to_bin(angle_rate_of_change, angle_rate_bins)])
        print("Episode - ", i_episode)
        done = False
        t = 0
        while not done:
            action = qlearn.chooseAction(state)
            # observation, reward, done, info = env.step(action)
            return_val = env.step(action)
            observation = return_val[0]
            reward = return_val[1]
            done = return_val[2]
            info = return_val[3]
            if done:
                print("Fallen - ", done)
            cart_position = observation[0]
            pole_angle = observation[1]
            cart_velocity = observation[2]
            angle_rate_of_change = observation[3]
            nextState = build_state([to_bin(cart_position, cart_position_bins),
                             to_bin(pole_angle, pole_angle_bins),
                             to_bin(cart_velocity, cart_velocity_bins),
                             to_bin(angle_rate_of_change, angle_rate_bins)])
            next_action = qlearn.chooseAction(state)
            if not(done):                
                qlearn.learn(state, action, reward, nextState, next_action)
                state = nextState
            else:
                reward = -375
                qlearn.learn(state, action, reward, nextState, next_action)
                last_time_steps = numpy.append(last_time_steps, [int(t + 1)])
                break
            t += 1

    l = last_time_steps.tolist()
    l.sort()
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))
    env.close()
