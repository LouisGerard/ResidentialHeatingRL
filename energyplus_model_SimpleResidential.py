from rl_testbed_for_energyplus.gym_energyplus.envs.energyplus_model import EnergyPlusModel
from gym import spaces
import numpy as np


class EnergyPlusModelResidential(EnergyPlusModel):
    def setup_spaces(self):
        self.action_space = spaces.Box(low=np.array([0]),
                                       high=np.array([50]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-20.0, -20.0, -20.0, 0.0, -5.0]),
                                            high=np.array([50.0, 50.0, 50.0, 1000000000.0, 5.0]),
                                            dtype=np.float32)

    def set_raw_state(self, raw_state=None):
        if raw_state is not None:
            self.raw_state = raw_state
        else:
            self.raw_state = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    def compute_reward(self):
        return self.raw_state[4]

    def format_state(self, raw_state):
        return np.array(raw_state)

    def read_episode(self, ep):
        pass

    def plot_episode(self, ep):
        pass

    def dump_timesteps(self, log_dir='', csv_file='', **kwargs):
        pass

    def dump_episodes(self, log_dir='', csv_file='', **kwargs):
        pass
