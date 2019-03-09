from rl_testbed_for_energyplus.gym_energyplus.envs.energyplus_model import EnergyPlusModel
from gym import spaces
import numpy as np
import pandas as pd
import os


class EnergyPlusModelResidential(EnergyPlusModel):
    def __init__(self, model_file):
        self.action_space = None
        self.observation_space = None

        super().__init__(model_file)

        self.day_i = 0
        self.time_i = 1
        self.out_temp_i = 2
        self.out_hum_i = 3
        self.wind_speed_i = 4
        self.wind_angle_i = 5
        self.rain_i = 6
        self.in_temp_i = 7
        self.in_hum_i = 8
        self.consumption_i = 9
        self.pmv_i = 10
        self.presence_i = 11

        self.df = None

    def setup_spaces(self):
        self.action_space = spaces.Box(low=np.array([-1]),
                                       high=np.array([1]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0, -22.8, 0.0, 0.0, 0.0, 0, -20.0, 0.0, 0.0]),
                                            high=np.array([6, 24, 35.0, 100.0, 15.4, 360.0, 1, 40.0, 100.0, 1]),
                                            dtype=np.float32)

    def set_raw_state(self, raw_state=None):
        if raw_state is not None:
            self.raw_state = raw_state

            # time
            if raw_state[self.time_i] != 24:
                self.raw_state[self.day_i] -= 1
            else:
                raw_state[self.time_i] = 0
                self.raw_state[self.day_i] = self.raw_state[self.day_i] % 7

            # wind angle
            self.raw_state[self.wind_angle_i] = np.floor(self.raw_state[self.wind_angle_i] / 90)

            # consumption
            if self.raw_state[self.consumption_i] != 0:
                self.raw_state[self.consumption_i] = 1
        else:
            self.raw_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def compute_reward(self):
        return self._compute_reward(self.raw_state[self.consumption_i],
                                    self.raw_state[self.pmv_i],
                                    self.raw_state[self.presence_i])

    def _compute_reward(self, cons, comf, presence):
        comf = np.abs(comf) * presence
        # return comf
        if comf < 0.5:
            comf = 0
        else:
            comf = (comf * 2) ** 2
        return comf
        return -cons / 300000 + comf

    def format_state(self, raw_state):
        return np.array(raw_state[:-2])

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

        self.df = pd.read_csv(file_path)

        self.rewards = []
        for cons, comf, pres in zip(
                self.df['ZONE ONE BASEBOARD HEAT:Baseboard Electric Power [W](TimeStep)'],
                self.df['ZONE ONE PEOPLE:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'],
                self.df['PRESENCE SCH:Schedule Value [](TimeStep)']
        ):
            self.rewards.append(self._compute_reward(cons, float(comf), pres))

    def highlight(self, indices, ax):
        i = 0
        while i < len(indices):
            ax.axvspan(indices[i] - 0.5, indices[i] + 0.5, facecolor='pink', edgecolor='none', alpha=.7)
            i += 1

    def plot_episode(self, ep):
        self.read_episode(ep)

        self.num_axes = 5
        self.axepisode = []
        for i in range(self.num_axes):
            if i == 0:
                ax = self.fig.add_axes([0.05, 1.00 - 0.70 / self.num_axes * (i + 1), 0.90, 0.7 / self.num_axes * 0.85])
            else:
                ax = self.fig.add_axes([0.05, 1.00 - 0.70 / self.num_axes * (i + 1), 0.90, 0.7 / self.num_axes * 0.85], sharex=self.axepisode[0])
            ax.set_xmargin(0)
            self.axepisode.append(ax)
            # ax.set_xticks(self.x_pos)
            # ax.set_xticklabels(self.x_labels)
            # ax.tick_params(labelbottom='off')
            ax.grid(True)

        idx = 0

        # Plot zone and outdoor temperature
        ax = self.axepisode[idx]
        idx += 1
        ax.lines = []
        self.highlight(self.df.loc[self.df['Environment:Site Rain Status [](TimeStep)'] == 1].index, ax)
        ax.plot(self.df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'],
                label='out')
        ax.plot(self.df['ZONE ONE PEOPLE:Zone Thermal Comfort Operative Temperature [C](TimeStep)'],
                alpha=0.5,
                c='orange',
                label='in')
        ax.legend()
        ax.set_ylabel('Temperature (C)')

        ax = self.axepisode[idx]
        idx += 1
        ax.plot(self.df['Environment:Site Outdoor Air Relative Humidity [%](TimeStep)'],
                label='out')
        ax.plot(self.df['ZONE ONE PEOPLE:People Air Relative Humidity [%](TimeStep)'],
                alpha=0.5,
                c='orange',
                label='in')
        ax.legend()
        ax.set_ylabel('Humidity (%)')

        ax = self.axepisode[idx]
        idx += 1
        ax.plot(self.df['ZONE ONE BASEBOARD HEAT:Baseboard Electric Power [W](TimeStep)'] != 0,
                label='consumption')
        ax.plot(self.df['ZONE ONE PEOPLE:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'],
                alpha=0.5,
                c='orange',
                label='comfort')
        ax.plot(np.array(self.rewards) / 2,
                alpha=0.5,
                c='green',
                label='reward')
        ax.legend()
        ax.set_ylabel('Reward')

        ax = self.axepisode[idx]
        idx += 1
        ax.plot(self.df['CLOTHING SCH:Schedule Value [](TimeStep)'],
                label='clothing')
        ax.plot(self.df['ACTIVITY SCH:Schedule Value [](TimeStep)'] / 104,
                alpha=0.5,
                c='orange',
                label='activity')
        ax.plot(self.df['PRESENCE SCH:Schedule Value [](TimeStep)'],
                alpha=0.5,
                c='green',
                label='presence')
        ax.legend()
        ax.set_ylabel('User')

        print('Mean discomfort :',
              (self.df['ZONE ONE PEOPLE:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'] *
               self.df['PRESENCE SCH:Schedule Value [](TimeStep)']).
              abs().mean())
        print('Mean consumption :',
              self.df['ZONE ONE BASEBOARD HEAT:Baseboard Electric Power [W](TimeStep)'].mean())
        print('Mean reward :',
              np.mean(self.rewards))

    def dump_timesteps(self, log_dir='', csv_file='', **kwargs):
        pass  # todo dump_timesteps

    def dump_episodes(self, log_dir='', csv_file='', **kwargs):
        pass  # todo dump_episodes

    def get_time(self):
        return int(self.raw_state[self.day_i]), self.raw_state[self.time_i]
