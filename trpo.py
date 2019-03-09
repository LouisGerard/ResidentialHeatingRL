#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines_energyplus.common.energyplus_util import make_energyplus_env, energyplus_logbase_dir
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
import os
import sys
import datetime
import main


def train(env_id, num_timesteps, seed):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         hid_size=32, num_hid_layers=2)

    # Create a new base directory like /tmp/openai-2018-05-21-12-27-22-552435
    global log_dir
    log_dir = os.path.join(energyplus_logbase_dir(), datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    if not os.path.exists(log_dir + '/output'):
        os.makedirs(log_dir + '/output')
    os.environ["ENERGYPLUS_LOG"] = log_dir
    model = os.getenv('ENERGYPLUS_MODEL')
    if model is None:
        print('Environment variable ENERGYPLUS_MODEL is not defined')
        sys.exit()
    weather = os.getenv('ENERGYPLUS_WEATHER')
    if weather is None:
        print('Environment variable ENERGYPLUS_WEATHER is not defined')
        sys.exit()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        print('train: init logger with dir={}'.format(log_dir))  # XXX
        logger.configure(log_dir)
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)

    global env
    env = make_energyplus_env(env_id, workerseed)

    trpo_mpi.learn(env, policy_fn,
                   # callback=plot,
                   max_timesteps=num_timesteps,
                   timesteps_per_batch=16*1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                   gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    env.close()


last_episode = 0


def plot(*args):
    global last_episode
    if args[0]['episodes_so_far'] > last_episode:
        env.env.plot(csv_file=log_dir + '/output/episode-%08d/eplusout.csv.gz' % last_episode)
        last_episode = args[0]['episodes_so_far']


if __name__ == '__main__':
    train('EnergyPlus-v0', num_timesteps=100000000, seed=12)
