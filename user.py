import numpy as np
from typing import List, Any
from distributions import read_distribution


class Activity:
    def __init__(self,
                 name: str,
                 distribution_file: str,
                 duration: float,
                 duration_std: float,
                 clothes: float = 0.5,
                 clothes_std: float = 0.1,
                 metabolic: float = 1.0,
                 metabolic_std: float = 0.05,
                 delay: float = 0.0,
                 delay_std: float = 0.0,
                 hunger_influence: float = 0.6,
                 energy_influence: float = 0.9,
                 hygiene_influence: float = 0.9,
                 hunger_restore: bool = False,
                 energy_restore: bool = False,
                 hygiene_restore: bool = False,
                 presence: bool = True) -> None:
        """

        :param name: Printed name of the activity
        :param distribution_file: Path to the distribution CSV
        :param duration: Mean duration in hours
        :param duration_std: Standard duration deviation in hours
        :param clothes: Mean clothing insulation
        :param clothes_std: Standard clothing insulation deviation
        :param delay: Mean preparing time in hours (optional)
        :param delay_std: Standard preparing time deviation in hours (optional)
        :param hunger_influence: Activity impact on hunger (optional)
        :param energy_influence: Activity impact on energy (optional)
        :param hygiene_influence: Activity impact on hygiene (optional)
        """
        self.name = name
        self.distribution = read_distribution(distribution_file)

        self.delay = np.random.normal(delay, delay_std)
        self.duration = np.random.normal(duration, duration_std)
        self.clothes = np.random.normal(clothes, clothes_std)
        self.metabolic = np.random.normal(metabolic, metabolic_std)

        self.duration_std = duration_std  # instance noises, other parameters ?
        self.delay_std = delay_std
        # self.clothes_std = clothes_std
        # self.metabolic_std = metabolic_std

        self.hunger_influence = hunger_influence
        self.energy_influence = energy_influence
        self.hygiene_influence = hygiene_influence

        self.hunger_restore = hunger_restore
        self.energy_restore = energy_restore
        self.hygiene_restore = hygiene_restore

        self.presence = presence

        self.started_day = None
        self.started_time = None
        self.current_duration = None
        self.current_delay = None

    def p_start(self, day, time, user):
        return self.distribution[day][int(time)]

    def step(self,
             day: int,
             time: float,
             user: Any) -> bool:  # FIXME can't hint User class (codependency)
        # startup
        user.current_activity = self
        if self.started_day is None:
            self.started_day = day
            self.started_time = time
            self.current_duration = np.random.normal(self.duration, self.duration_std)
            self.current_delay = np.random.normal(self.delay, self.delay_std)

        if day < self.started_day:
            day += 7
        time_offset = (day - self.started_day) * 24
        relative_time = time - self.started_time + time_offset

        if relative_time < self.current_delay:
            return False

        # presence
        user.presence = self.presence

        user.clothes = self.clothes
        user.metabolic = self.metabolic

        if relative_time < self.current_duration:
            return False

        user.presence = True

        user.current_hunger *= self.hunger_influence
        user.current_energy *= self.energy_influence
        user.current_hygiene *= self.hygiene_influence

        if self.hunger_restore:
            user.current_hunger = user.base_hunger

        if self.energy_restore:
            user.current_energy = user.base_energy

        if self.hygiene_restore:
            user.current_hygiene = user.base_hygiene

        return True

    def restart(self):
        self.__init__()  # FIXME

    def __repr__(self) -> str:
        return self.name


class Sport(Activity):
    def __init__(self):
        Activity.__init__(self,
                          name='Sport',
                          distribution_file='data/sport_distribution.csv',
                          duration=1.5,
                          duration_std=0.25,
                          clothes=0.4,
                          clothes_std=0.05,
                          metabolic=3.0,
                          metabolic_std=0.5,
                          delay=5/60,
                          delay_std=1/60,
                          hunger_influence=0.3,
                          energy_influence=0.4,
                          hygiene_influence=0.05)

    def p_start(self, day, time, user):
        # account user energy
        return Activity.p_start(self, day, time, user) * min(user.current_energy, 1)


class Sleep(Activity):  # todo get naked
    def __init__(self):
        Activity.__init__(self,
                          name='Sleep',
                          distribution_file='data/sleep_distribution.csv',
                          duration=7.5,
                          duration_std=1,
                          clothes=1.0,
                          clothes_std=0.25,
                          metabolic=0.8,
                          delay=10/60,
                          delay_std=5/60,
                          energy_restore=True)

    def p_start(self, day, time, user):
        # account user energy
        return Activity.p_start(self, day, time, user) / user.current_energy


class GoOut(Activity):
    def __init__(self):
        Activity.__init__(self,
                          name='Go out',
                          distribution_file='data/outside_distribution.csv',
                          duration=5,
                          duration_std=2,
                          clothes=1.0,
                          clothes_std=0.25,
                          metabolic=1.5,
                          metabolic_std=0.25,
                          delay=10/60,
                          delay_std=2/60,
                          presence=False)

        def step(self,
                 day: int,
                 time: float,
                 user: Any) -> bool:
            # random outside eat
            if Activity.step(self, day, time, user):
                if np.random.rand() > 0.3:
                    user.current_hunger = user.base_hunger
                return True
            return False


class Eat(Activity):  # todo cook
    def __init__(self):
        Activity.__init__(self,
                          name='Eat',
                          distribution_file='data/eat_distribution.csv',
                          duration=1,
                          duration_std=1/2,
                          delay=10/60,
                          delay_std=2/60,
                          hunger_restore=True)

    def p_start(self, day, time, user):
        # account user hunger
        return Activity.p_start(self, day, time, user) / user.current_hunger


class Shower(Activity):  # todo get naked
    def __init__(self):
        Activity.__init__(self,
                          name='Shower',
                          distribution_file='data/shower_distribution.csv',
                          duration=10/60,
                          duration_std=5/60,
                          metabolic=1.5,
                          metabolic_std=0.25,
                          delay=10/60,
                          delay_std=5/60,
                          hygiene_restore=True)

    def p_start(self, day, time, user):
        # account user hygiene
        if user.current_hygiene >= 0.6:
            return 0
        return Activity.p_start(self, day, time, user) / user.current_hygiene


class TV(Activity):
    def __init__(self):
        Activity.__init__(self,
                          name='TV',
                          distribution_file='data/tv_distribution.csv',
                          duration=2,
                          duration_std=1/2)


class User:
    def __init__(self,
                 preference: float,
                 sensitivity: float,
                 activities: List[Activity]) -> None:
        self.preference = preference  # todo preferences & sensitivity
        self.sensitivity = sensitivity
        self.activities = activities

        self.presence = True
        self.clothes = 0.5
        self.metabolic = 1.0
        self.current_activity = None

        self.base_hunger = np.random.normal(1.0, 0.1)
        self.base_energy = np.random.normal(1.0, 0.1)
        self.base_hygiene = np.random.normal(1.0, 0.1)
        self.current_hunger = self.base_hunger
        self.current_energy = self.base_energy
        self.current_hygiene = self.base_hygiene

    def choose_activity(self, day, time):
        sum_p = 0
        p = []
        for a in self.activities:
            p.append(a.p_start(day, time, self))
            sum_p += p[-1]
        p = np.array(p) / sum_p
        self.current_activity = np.random.choice(self.activities, p=p)
        return self.current_activity


if __name__ == '__main__':
    activities = [
        Sport(),
        Sleep(),
        GoOut(),
        Shower(),
        Eat(),
        TV()
    ]
    dana = User(0.1, 0.1, activities)

    day = 0
    time = 0
    dana.current_energy = 0.1
    dana.choose_activity(day, time)

    from time import sleep

    while True:
        if dana.current_activity.step(day, time, dana):
            dana.current_activity.restart()
            dana.choose_activity(day, time)

            # print('-' * 50)
            # print('day', day, 'time', time)
            # print('Dana is doing', dana.current_activity.name)
            # print('hunger', dana.current_hunger)
            # print('energy', dana.current_energy)
            # print('hygiene', dana.current_hygiene)
            #
            # for a in activities:
            #     print(a.name, end='\t')
            # print()
            #
            # sleep(1)

        time += 0.5
        if time >= 24:
            time = time % 24
            day = (day + 1) % 7
