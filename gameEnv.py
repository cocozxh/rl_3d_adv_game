from cmath import inf
from xmlrpc.client import boolean
import numpy as np
import gym
from gym import spaces
from learner import Learner
from generator import Generator
import torch
import functools
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec

NUM_ITERS = 100

def env():
    '''
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = raw_env()
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # # this wrapper helps error handling for discrete action spaces
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env():
    '''
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    '''
    env = gameEnv()
    env = parallel_to_aec(env)
    return env

class gameEnv(ParallelEnv):

    metadata = {'render_modes': ['human'], "name": "rps_v2"}

    def __init__(self) -> None:
        torch.cuda.empty_cache()

        import argparse

        parser = argparse.ArgumentParser()

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        parser.add_argument('--data_path', type=str, default='data')
        parser.add_argument('--mesh_dir', type=str, default='data/meshes')
        parser.add_argument('--bg_dir', type=str, default='data/background')
        parser.add_argument('--test_bg_dir', type=str, default='data/test_background')
        parser.add_argument('--output', type=str, default='out/patch')

        parser.add_argument('--patch_num', type=int, default=1)
        parser.add_argument('--patch_dir', type=str, default='example_logos/fasterrcnn_chest_legs')
        parser.add_argument('--idx', type=str, default='idx/chest_legs1.idx')

        parser.add_argument('--iterations', type=int, default=5)
        parser.add_argument('--learner_epochs', type=int, default=1)
        parser.add_argument('--learner_update', action='store_true')
        parser.add_argument('--generator_epochs', type=int, default=20)

        parser.add_argument('--img_size', type=int, default=416)
        parser.add_argument('--num_bgs', type=int, default=10)
        parser.add_argument('--num_test_bgs', type=int, default=2)
        parser.add_argument('--num_angles_test', type=int, default=1)
        parser.add_argument('--angle_range_test', type=int, default=0)
        parser.add_argument('--num_angles_train', type=int, default=1)
        parser.add_argument('--angle_range_train', type=int, default=0)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--rand_translation', type=int, default=50)
        parser.add_argument('--num_meshes', type=int, default=1)

        parser.add_argument('--cfgfile', type=str, default="cfg/yolo.cfg")
        parser.add_argument('--weightfile', type=str, default="data/yolov2/yolo.weights")

        parser.add_argument('--detector', type=str, default='yolov2')
        parser.add_argument('--test_only', action='store_true')

        parser.add_argument('--log_dir', type=str, default='log/0415/')

        self.config = parser.parse_args()

        self.step_count = 0
        self.learner = Learner(self.config, device)
        self.generator = Generator(self.config, device)

        self.possible_agents = ["learner", "generator"]

        # self.observation_spaces = spaces.Dict({'generator':spaces.Box(low=np.array([0]), high=np.array([1])), 'learner':spaces.Box(low=np.array([0]), high=np.array([1]))})
        
        # generator_action_space = spaces.Dict({name:spaces.Box(low=-inf, high=inf, shape=params.shape) for name, params in self.generator.G.named_parameters()})
        
        # learner_action_space = spaces.Dict({name:spaces.Box(low=-inf, high=inf, shape=params.shape) for name, params in self.learner.dnet.named_parameters()})
        
        # self.action_spaces = spaces.Dict({'generator':generator_action_space, 'learner':learner_action_space})
        self.generator.initialize_patch()
        self.images = self.generator.create_images()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=np.array([0]), high=np.array([1]))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if agent=='generator':
            return spaces.Dict({name:spaces.Box(low=-inf, high=inf, shape=params.shape) for name, params in self.generator.G.named_parameters()})
        elif agent=='learner':
            return spaces.Dict({name:spaces.Box(low=-inf, high=inf, shape=params.shape) for name, params in self.learner.dnet.named_parameters()})

    def step(self, actions):
        self.step_count += 1
        
        #generator update with action1
        # print(actions['generator'])
        self.generator.update_params(actions['generator'])
        self.images = self.generator.create_images()

        #learner update with action2
        self.learner.update_params(actions['learner'])
        loss,acc = self.learner.evaluate(self.images)

        self.observations = {agent: acc.item() for agent in self.agents}

        self.rewards = {}
        self.rewards[self.agents[0]] = acc.item()
        self.rewards[self.agents[1]] = loss.item()

        env_done = self.step_count >= NUM_ITERS
        self.dones = {agent: env_done for agent in self.agents}

        # self.dones = self.check_done(obs)

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        self.infos = {agent: {} for agent in self.agents}

        if env_done:
            self.agents = []

        return self.observations, self.rewards, self.dones, self.infos

    def render(self, mode='human'):
        '''
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        '''
        if len(self.agents) == 2:
            string = ("Current accuracy: {}".format(self.observations['learner']))
        else:
            string = "Game over"
        print(string)

    def reset(self):
        self.agents = self.possible_agents[:]
        self.learner.reset()
        self.generator.reset()
        loss,acc = self.learner.evaluate(self.images)
        observations = {agent: np.array([np.float32(loss.item())]) for agent in self.agents}

        # loss,acc = self.learner.evaluate(self.images)
        # obs = loss
        # self.dones = {'learner':False,'generator':False}

        return observations

    # def check_done(self, obs) -> boolean:
    #     if self.step_count==10000:
    #         return {'learner':True,'generator':True}
    #     elif(obs <= 0.1): #accuracy is good
    #         return {'learner':True,'generator':True}
    #     else:
    #         return {'learner':False,'generator':False}


