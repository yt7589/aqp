#
import unittest
import gym
from app.drl.cart_pole_model import CartPoleModel

class TCartPoleModel(unittest.TestCase):
    def test_f001(self):
        env = gym.make('CartPole-v0')
        model = CartPoleModel(num_actions=env.action_space.n)
        obs = env.reset()
        # Cart position, Cart Velocity, pole angle, pole velocity at tip
        print('type:{0}; shape:{1}'.format(type(obs), obs.shape))
        print(obs)
        action, value = model.action_value(obs[None, :])
        # action: 0-Left; 1-Right
        print(action, value) # [1] [-0.00145713]
        self.assertTrue(True)



        