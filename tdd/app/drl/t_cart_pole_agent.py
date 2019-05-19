import unittest
import pickle
import gym
import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import matplotlib
import matplotlib.pyplot as plt
from app.drl.cart_pole_model import CartPoleModel
from app.drl.cart_pole_agent import CartPoleAgent

class TCartPoleAgent(unittest.TestCase):
    def test_001(self):
        env = gym.make('CartPole-v0')
        model = CartPoleModel(num_actions=env.action_space.n)
        agent = CartPoleAgent(model)
        rewards_sum = agent.test(env, render=True)
        print("Total Episode Reward: %d out of 200" % rewards_sum)
        self.assertTrue(True)

    def test_002(self):
        model_file = './work/cartpole.a2c'
        model_file_trained = './work/cartpole_v1.a2c'
        env = gym.make('CartPole-v0')
        model = CartPoleModel(num_actions=env.action_space.n)
        model.load_weights(model_file)
        agent = CartPoleAgent(model)
        logging.getLogger().setLevel(logging.INFO)
        rewards_history = agent.train(env)
        print("Finished training.")
        print("Total Episode Reward: %d out of 200" % agent.test(env, render=True))
        plt.style.use('seaborn')
        plt.plot(np.arange(0, len(rewards_history), 5), rewards_history[::5])
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.show()
        model.save_weights(model_file_trained)
        self.assertTrue(True)