import gym
from stable_baselines3 import PPO
import singleAgentGameEnvironment
import warnings
import torch
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


import argparse

parser = argparse.ArgumentParser(description='learner and generator parameter')

parser.add_argument('--without_mesh', action='store_false')
parser.add_argument('--generator_model', type=str, default='mlp')
parser.add_argument('--data_path', type=str, default='data')
parser.add_argument('--mesh_dir', type=str, default='data/meshes')
parser.add_argument('--bg_dir', type=str, default='data/background')
parser.add_argument('--test_bg_dir', type=str, default='data/test_background')
parser.add_argument('--output', type=str, default='out/patch')

parser.add_argument('--patch_num', type=int, default=1)
parser.add_argument('--patch_dir', type=str, default='example_logos/fasterrcnn_chest_legs')
parser.add_argument('--idx', type=str, default='idx/chest_legs1.idx')

parser.add_argument('--iterations', type=int, default=5)
parser.add_argument('--learner_update_epochs', type=int, default=1)
parser.add_argument('--generator_epochs', type=int, default=20)

parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--num_bgs', type=int, default=5)
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

config = parser.parse_args()

env = singleAgentGameEnvironment.create_env(config, device)

torch.cuda.empty_cache()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./single_agent_tensorboard/", n_steps=2, batch_size=2)

torch.cuda.empty_cache()

print("Start Learning!")
model.learn(total_timesteps=1)
model.save("singleAgent")

print("Model Saved!")

model = PPO.load("singleAgent")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()