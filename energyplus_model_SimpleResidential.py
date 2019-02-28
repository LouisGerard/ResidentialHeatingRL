from rl_testbed_for_energyplus.gym_energyplus.envs.energyplus_model import EnergyPlusModel
from gym import spaces
import numpy as np
import pandas as pd
import os


class EnergyPlusModelResidential(EnergyPlusModel):
    def setup_spaces(self):
        self.action_space = spaces.Box(low=np.array([-50, 0, 0, 0]),
                                       high=np.array([50, 5, 900, 1]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0, -22.8, 0.0, 0.0, 0.0, 0, -20.0, 0.0, -5.0]),
                                            high=np.array([6, 24, 35.0, 100.0, 15.4, 360.0, 1, 40.0, 3500, 5.0]),
                                            dtype=np.float32)

    def set_raw_state(self, raw_state=None):
        if raw_state is not None:
            self.raw_state = raw_state
            if raw_state[1] != 24:
                self.raw_state[0] -= 1
            else:
                raw_state[1] = 0
        else:
            self.raw_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def compute_reward(self):
        return self.raw_state[8] + self.raw_state[9]

    def format_state(self, raw_state):
        return np.array(raw_state)

    def read_episode(self, ep):
        file_path = ''
        if type(ep) is str:
            file_path = ep
        else:
            ep_dir = self.episode_dirs[ep]
            for file in ['eplusout.csv', 'eplusout.csv.gz']:
                file_path = ep_dir + '/' + file
                if os.path.exists(file_path):
                    break
            else:
                print('No CSV or CSV.gz found under {}'.format(ep_dir))
                quit()

        df = pd.read_csv(file_path)

    def plot_episode(self, ep):
        pass  # todo plot_episode

    def dump_timesteps(self, log_dir='', csv_file='', **kwargs):
        pass  # todo dump_timesteps

    def dump_episodes(self, log_dir='', csv_file='', **kwargs):
        pass  # todo dump_episodes

    def get_time(self):
        return int(self.raw_state[0]), self.raw_state[1]
