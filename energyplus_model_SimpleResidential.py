from rl_testbed_for_energyplus.gym_energyplus.envs.energyplus_model import EnergyPlusModel
from gym import spaces
import numpy as np
import pandas as pd
import os


class EnergyPlusModelResidential(EnergyPlusModel):
    def __init__(self, model_file, user):
        self.user = user

        self.day_i = 0
        self.time_i = 1
        self.out_temp_i = 2
        self.out_hum_i = 3
        self.in_temp_i = 4
        self.in_hum_i = 5
        self.consumption_i = 6
        self.pmv_i = 7
        self.presence_i = 8
        self.clothes_i = 9
        self.metabolic_i = 10

        self.df = None

        self.factor = 2

        self.state_composition = {
            self.out_temp_i: False,
            self.out_hum_i: False,
            self.in_temp_i: False,
            self.in_hum_i: False,
            self.clothes_i: False,
            self.metabolic_i: False,
            self.pmv_i: True,
            self.presence_i: True,
        }

        self.state_spaces = {
            self.out_temp_i: (-22.8, 40),
            self.out_hum_i: (0, 100),
            self.in_temp_i: (-20, 40),
            self.in_hum_i: (0, 100),
            self.clothes_i: (0, 5),
            self.metabolic_i: (0, 5),
            self.pmv_i: (0, 5),
            self.presence_i: (0, 1),
        }

        super().__init__(model_file)

    def setup_spaces(self):
        self.action_space = spaces.Box(low=np.array([-1]),
                                       high=np.array([1]),
                                       dtype=np.float32)

        low = []
        high = []
        for s_in, (l, h) in zip(self.state_composition.values(), self.state_spaces.values()):
            if s_in:
                low.append(l)
                high.append(h)

        self.observation_space = spaces.Box(low=np.array(low),
                                            high=np.array(high),
                                            dtype=np.float32)

    def set_raw_state(self, raw_state=None):
        if raw_state is not None:
            self.raw_state = raw_state.tolist()

            # time
            if raw_state[self.time_i] != 24:
                self.raw_state[self.day_i] -= 1
            else:
                self.raw_state[self.time_i] = 0
                self.raw_state[self.day_i] = self.raw_state[self.day_i] % 7

            # consumption
            if self.raw_state[self.consumption_i] != 0:
                self.raw_state[self.consumption_i] = 1
        else:
            self.raw_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.raw_state.append(self.user.clothes)
        self.raw_state.append(self.user.metabolic)

    def compute_reward(self):
        return self._compute_reward(self.raw_state[self.consumption_i],
                                    self.raw_state[self.pmv_i],
                                    self.raw_state[self.presence_i])

    def _compute_reward(self, cons, comf, presence):
        # return -cons
        comf = np.abs(comf) * presence
        # return -comf
        return -cons - comf*self.factor

    def format_state(self, raw_state):
        state = []
        ob_i = 0
        for i, s_in in self.state_composition.items():
            if s_in:
                s = raw_state[i]
                s /= (self.observation_space.high[ob_i] - self.observation_space.low[ob_i]) / 2
                s -= 1
                state.append(s)
                ob_i += 1

        return np.array(state)

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
            self.rewards.append(self._compute_reward(cons/3000, float(comf), pres))

    def highlight(self, indices, ax):
        i = 0
        while i < len(indices):
            ax.axvspan(indices[i] - 0.5, indices[i] + 0.5, facecolor='pink', edgecolor='none', alpha=.7)
            i += 1

    def show_diff(self, ep=0):
        if self.df is None:
            self.read_episode(ep)

        result = {}
        on = self.df[self.df['ZONE ONE BASEBOARD HEAT:Baseboard Electric Power [W](TimeStep)'] != 0]
        off = self.df[self.df['ZONE ONE BASEBOARD HEAT:Baseboard Electric Power [W](TimeStep)'] == 0]
        stats = {
            'Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)': np.mean,
            'Environment:Site Outdoor Air Relative Humidity [%](TimeStep)': np.mean,
            'ZONE ONE PEOPLE:People Air Relative Humidity [%](TimeStep)': np.mean,
            'ZONE ONE PEOPLE:Zone Thermal Comfort Fanger Model PMV [](TimeStep)': lambda x: x.abs().mean(),
            'ZONE ONE PEOPLE:Zone Thermal Comfort Operative Temperature [C](TimeStep)': np.mean,
            'PRESENCE SCH:Schedule Value [](TimeStep)': np.mean,
            'ACTIVITY SCH:Schedule Value [](TimeStep)': np.mean,
            'CLOTHING SCH:Schedule Value [](TimeStep)': np.mean,
        }
        for col in stats.keys():
            on_mean = stats[col](on[col])
            off_mean = stats[col](off[col])
            result[col] = [on_mean, off_mean]
        return pd.DataFrame.from_dict(result).T

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
        ax.plot(self.df['ZONE ONE BASEBOARD HEAT:Baseboard Electric Power [W](TimeStep)'] / 1500 - 1,
                label='consumption')
        ax.plot(self.df['ZONE ONE PEOPLE:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'],
                alpha=0.5,
                c='orange',
                label='comfort')
        ax.plot(self.df['ZONE ONE PEOPLE:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'] *
                self.df['PRESENCE SCH:Schedule Value [](TimeStep)'],
                alpha=0.5,
                c='black',
                label='comfort*presence')
        ax.legend()
        ax.set_ylabel('Objectives')

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

        ax = self.axepisode[idx]
        idx += 1
        ax.plot(np.array(self.rewards) / 2,
                alpha=0.5,
                c='green',
                label='reward')
        ax.legend()
        ax.set_ylabel('Reward')

        print('Mean discomfort :',
              (self.df['ZONE ONE PEOPLE:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'] *
               self.df['PRESENCE SCH:Schedule Value [](TimeStep)']).
              abs().mean())
        print('Mean consumption :',
              self.df['ZONE ONE BASEBOARD HEAT:Baseboard Electric Power [W](TimeStep)'].mean())
        print('Mean reward :',
              np.mean(self.rewards))
        print('Mean presence :',
              self.df['PRESENCE SCH:Schedule Value [](TimeStep)'].mean())

    def dump_timesteps(self, log_dir='', csv_file='', **kwargs):
        pass  # todo dump_timesteps

    def dump_episodes(self, log_dir='', csv_file='', **kwargs):
        pass  # todo dump_episodes

    def get_time(self):
        return int(self.raw_state[self.day_i]), self.raw_state[self.time_i]
