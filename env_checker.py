import gameEnv
from pettingzoo.test import parallel_api_test
from pettingzoo.test.seed_test import parallel_seed_test
from gameEnv import gameEnv
import singleAgentGameEnvironment
from stable_baselines3.common.env_checker import check_env

# Parallel API test
# env = gameEnv()
# parallel_api_test(env, num_cycles=1)

# single Agent Environment Test
env = singleAgentGameEnvironment.create_env()
check_env(env)

