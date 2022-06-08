from cmath import inf
from distutils.command.config import config
import numpy as np
import gym
from gym import spaces
from learner import Learner
from generator import Generator
from generator_without_mesh import Generator_without_Mesh
import torch
from flatten_action import FlattenAction

NUM_ITERS = 10

def create_env(config, device):

    env = singleAgentGameEnvironment(config, device)
    
    # UserWarning: The action space is based off a Dict. This type of action space is currently not supported by Stable Baselines 3. You should try to flatten the action using a wrapper.

    env = FlattenAction(env)

    return env

class singleAgentGameEnvironment(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(self, config, device) -> None:

        self.config = config
        self.step_count = 0
        self.learner = Learner(self.config, device)

        if self.config.without_mesh:
            self.generator = Generator(self.config, device)
            print("Generator use mesh!")
        else:
            self.generator = Generator_without_Mesh(self.config, device)
            print("Generator do not use mesh!")
        
        # generator will initialize patch when creating new generator object
        # self.generator.initialize_patch()
        self.images = self.generator.create_images()

        self.observation_shape = (1,)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape))

        self.action_space = spaces.Dict({name:spaces.Box(low=-1, high=1, shape=params.shape) for name, params in self.generator.G.named_parameters()})
    
    def step(self, action):
        self.step_count += 1
        #generator update with action1
        # print(actions['generator'])
        self.generator.update_patch(action)
        
        self.images = self.generator.create_images()

        for _ in range(self.config.learner_update_epochs):
            self.learner.update(self.images)
        loss,acc = self.learner.evaluate(self.images)

        self.observation = acc.item()
        self.reward = loss.item()

        env_done = self.step_count >= NUM_ITERS
        self.done = env_done

        # self.done = self.check_done(obs)

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        self.info = {}

        return np.array([np.float32(self.observation)]), self.reward, self.done, self.info

    def render(self, mode='human'):
        '''
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        '''
        string = ("Current accuracy: {}".format(self.observation))
        print(string)

    def reset(self):
        self.learner.reset()
        self.generator.reset()
        loss,acc = self.learner.evaluate(self.images)
        observation = np.float32(loss.item())
        self.done = False
        # loss,acc = self.learner.evaluate(self.images)
        # obs = loss
        # self.dones = {'learner':False,'generator':False}

        return np.array([observation])

    # def check_done(self, obs) -> boolean:
    #     if self.step_count==10000:
    #         return {'learner':True,'generator':True}
    #     elif(obs <= 0.1): #accuracy is good
    #         return {'learner':True,'generator':True}
    #     else:
    #         return {'learner':False,'generator':False}


