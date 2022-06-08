# Reinforcement Learning 3D Adversarial Game
## Prerequisites
Our human meshes, background images, and yolov2 model file can be downloaded from:
https://mega.nz/file/pZtUCKza#6AF3AkIYxiWXysqoo78nbjKoTCos6-PwU_UBaSntIA8

After extracting the .zip file, your directory should contain
```
./data/background
./data/meshes
./data/test_background
./data/yolov2
```

PyTorch 1.8.0 and Torchvision 0.8.1 (tested in python 3.8):
```
pip install torch torchvision
```
PyTorch3D v0.2.5:
```
pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.2.5'
```
## Introduction
We build a single agent reinforcement learning environment with OpenAI Gym in _singleAgentGameEnvironment.py_.
The environment is checked by stablebaseline env checker.
The only rl agent is the generator, which will attach patches to 3D meshes, and will output 2D images after rendering.
Here, we use ResNet as the model of generator. The output of Resnet is patches.
The learner updates itself with gradient descent.
We train the single agent in this environment with API provided by stablebaseline.
However, most ResNet models make the training suffer from CUDA out of memory.
## Training
To train the agent
``` python singleAgentTestModel.py ```
You can also specify parameters provided in that file by modifing the commands.
By default, the model trained with PPO.
