import pandas as pd

import main
import numpy as np
import keras
import datetime
from baselines_energyplus.common.energyplus_util import make_energyplus_env, energyplus_logbase_dir
import os
import sys
from baselines import logger


class Agent:
    def __init__(self, env_id, seed):
        self.env_id = env_id
        self.gamma = 0.99  # fixme
        self.epsilon = 0.1
        self.epsilon_decay = 0.5  # 0.92
        self.seed = seed
        self.history = []

        # Create a new base directory like /tmp/openai-2018-05-21-12-27-22-552435
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

        print('train: init logger with dir={}'.format(log_dir))  # XXX
        logger.configure(log_dir)

        self.log_dir = log_dir

    def build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(32, activation='relu', input_dim=self.n_observations))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(2, activation='softmax'))

        model.compile(optimizer='sgd', loss='mse')
        return model

    def __enter__(self):
        self.env = make_energyplus_env(self.env_id, self.seed)
        self.env.env.ep_model.log_dir = self.log_dir
        self.n_observations = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.model = self.build_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.env.close()

    def train(self, num_episodes=None, exploit_mode=False):
        n_episode = 0
        while True:
            n_episode += 1
            if n_episode > num_episodes:
                break

            state = self.env.reset().reshape((1, -1))
            done = False

            print('EPISODE', n_episode, '; epsilon = %2f' % self.epsilon)
            print('-' * 50)

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(np.array([action * 2 - 1]))
                next_state = next_state.reshape((1, -1))

                if not exploit_mode:
                    self.fit_step(state, action, reward, next_state)
                state = next_state

            self.epsilon *= self.epsilon_decay

            self.env.env.ep_model.show_progress()

            print('Loss :', np.mean(self.history))
            self.history = []

            print(self.env.env.ep_model.show_diff(n_episode-1))
            print('-' * 50)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)
        return np.argmax(self.model.predict(state)[0])

    def fit_step(self, state, action, reward, next_state=None):
        predicted = self.model.predict(next_state)
        predicted[0][action] = reward + self.gamma * np.amax(self.model.predict(next_state))
        h = self.model.fit(state, predicted, epochs=1, verbose=0)
        self.history.append(h.history['loss'])


if __name__ == '__main__':

        # comforts = []
        # consumption = []
        # tested = np.arange(1.7, 2.4, 0.1)
        # i = 0
        #
        # for f in tested:
        #     i += 1
        #     with Agent('EnergyPlus-v0', 10) as agent:
        #         agent.model = keras.models.load_model('model%d.h5' % i)
        #         agent.epsilon = 0
        #         agent.train(1, exploit_mode=True)
        #
        #         comforts.append((agent.env.env.ep_model.df['ZONE ONE PEOPLE:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'].abs() * agent.env.env.ep_model.df['PRESENCE SCH:Schedule Value [](TimeStep)']).mean())
        #         consumption.append(agent.env.env.ep_model.df['ZONE ONE BASEBOARD HEAT:Baseboard Electric Power [W](TimeStep)'].mean() / 3000)
        #
        # import matplotlib.pyplot as plt
        # plt.plot(tested, comforts, label='Discomfort')
        # plt.plot(tested, consumption, label='Consumption')
        # plt.hlines(1.0, tested[0], tested[-1], linestyles='dashed')
        # plt.xlabel('Comfort factor')
        # plt.legend()
        # plt.show()

        # import itertools
        # lst = list(itertools.product([0, 1], repeat=4))
        # del lst[0]
        # lst = [
        #     [0, 0, 1, 0],
        #     [1, 0, 0, 1],
        #     [1, 0, 1, 1],
        # ]
        #
        # results = []
        # for i in range(4):
        #     results.append(pd.DataFrame([
        #             [[0, 0, 0] for _ in range(2)]
        #             for _ in range(2)
        #         ],
        #         columns=[0, 1]))
        #
        # for temp, hum, clothes, metabolic in lst:
        #     print('-' * 50)
        #     print('%d_%d_%d_%d' % (temp, hum, clothes, metabolic))
        #     print('-' * 50)
        #
        #     agent = Agent('EnergyPlus-v0', 10)
        #
        #     agent.env = make_energyplus_env(agent.env_id, agent.seed)
        #     agent.env.env.ep_model.log_dir = agent.log_dir
        #
        #     model = agent.env.env.ep_model
        #     model.state_composition[model.pmv_i] = False
        #     model.state_composition[model.in_temp_i] = bool(temp)
        #     model.state_composition[model.in_hum_i] = bool(hum)
        #     model.state_composition[model.clothes_i] = bool(clothes)
        #     model.state_composition[model.metabolic_i] = bool(metabolic)
        #     model.setup_spaces()
        #
        #     agent.n_observations = temp + hum + clothes + metabolic + 1
        #     agent.n_actions = agent.env.action_space.shape[0]
        #     agent.model = agent.build_model()
        #
        #     agent.model = keras.models.load_model('model_%d_%d_%d_%d.h5' % (temp, hum, clothes, metabolic))
        #     agent.epsilon = 0
        #     agent.train(1, exploit_mode=True)
        #
        #     plot_i = 2*clothes + metabolic
        #
        #     comfort = (agent.env.env.ep_model.df['ZONE ONE PEOPLE:Zone Thermal Comfort Fanger Model PMV [](TimeStep)'].abs() * agent.env.env.ep_model.df['PRESENCE SCH:Schedule Value [](TimeStep)']).mean()
        #     consumption = agent.env.env.ep_model.df['ZONE ONE BASEBOARD HEAT:Baseboard Electric Power [W](TimeStep)'].mean() / 3000
        #     reward = np.mean(agent.env.env.ep_model.rewards)
        #
        #     results[plot_i][temp][hum][0] = comfort
        #     results[plot_i][temp][hum][1] = consumption
        #     results[plot_i][temp][hum][2] = reward
        #
        #     agent.env.close()

            # agent.train(3)
            # agent.env.close()
            #
            # agent.model.save('model_%d_%d_%d_%d.h5' % (temp, hum, clothes, metabolic))

        # for df in results:
        #     print(df)

        with Agent('EnergyPlus-v0', 10) as agent:
            agent.model = keras.models.load_model('/home/louis/Documents/ResidentialHeatingRL/model4.h5')
            # agent.env.env.ep_model.factor = 3
            # try:
            #     agent.train(3)
            # except KeyboardInterrupt:
            #     pass
            # agent.model.save('model.h5')
        #
        #     # plot greedy
            agent.epsilon = 0
            agent.train(1, exploit_mode=True)
