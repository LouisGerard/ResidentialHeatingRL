import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def write_distribution(distribution: np.ndarray, filename: str) -> None:
    with open(filename, 'w') as f:
        for day_data in distribution:
            f.write(','.join(day_data.astype(str)) + '\n')


def normalize_distribution(distribution: np.ndarray):
    return distribution / distribution.sum()


def parse_sleep():
    # notes :
    # Counter({'Drank coffee': 524, 'Drank tea': 456, 'Worked out': 364, '': 235, 'Stressful day': 58, 'Ate late': 21})
    distribution = np.zeros((7, 24), dtype=int)
    sleep_amounts = []

    with open('data/sleepdata.csv', 'r') as f:
        # header
        f.readline()

        for line in f:
            if line == '\n':
                continue

            line = line.split(';')

            sleep_amount = line[3].split(':')
            sleep_amount = int(sleep_amount[0]) * 60 + int(sleep_amount[1])
            sleep_amounts.append(sleep_amount)

            start = datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S')
            end_sleep = start.hour + int(np.round((start.minute + sleep_amount) / 60))
            start_hour = start.hour
            start_day = start.weekday()
            while end_sleep > 0:
                distribution[start_day][start_hour:end_sleep] += 1
                start_day = (start_day + 1) % 7
                start_hour = 0
                end_sleep -= 24

    sleep_amounts = np.array(sleep_amounts)
    print('mean :', sleep_amounts.mean() / 60, 'std :', sleep_amounts.std() / 60)

    write_distribution(distribution, 'data/sleep_distribution.csv')


def parse_pedestrian():
    counts = np.zeros((7, 24), dtype=int)
    distribution = np.zeros((7, 24))
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    days = {k: i for i, k in enumerate(days)}

    with open('data/Pedestrian_volume_feb2019.csv', 'r') as f:
        # header
        f.readline()

        for line in f:
            line = line.split(',')
            day = days[line[5]]
            hour = int(line[6])
            traffic = int(line[9].split('.', 1)[0].rstrip('\n'))
            counts[day][hour] += 1
            distribution[day][hour] += (traffic - distribution[day][hour]) / counts[day][hour]

    write_distribution(distribution.astype(int), 'data/outside_distribution.csv')


def parse_paxraw_d():
    distribution = np.zeros((7, 24), dtype=int)

    with open('data/paxraw_d.csv', 'r') as f:
        for line in f:
            line = line.split(',')

            activity = int(line[7].split('.', 1)[0])

            day = int(line[3].split('.', 1)[0]) - 1
            hour = int(line[5].split('.', 1)[0])

            if activity > 4000:  # arbitrary threshold, mean above : 19877
                distribution[day][hour] += 1
    write_distribution(distribution, 'data/sport_distribution.csv')


def read_distribution(filename: str):
    result = []
    with open(filename, 'r') as f:
        for line in f:
            line = list(map(float, line.split(',')))
            result.append(line)
    return np.array(result)


def show_distribution(filename: str) -> None:
    x_plt = np.arange(24)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    with open(filename, 'r') as f:
        i = 0
        for line in f:
            line = list(map(float, line.split(',')))

            plt.plot(x_plt, line, label=days[i])

            i += 1

        plt.legend()
        plt.show()


if __name__ == '__main__':
    ds = [
        'data/sport_distribution.csv',
        'data/sleep_distribution.csv',
        'data/outside_distribution.csv',
        'data/eat_distribution.csv',
        'data/shower_distribution.csv',
        'data/tv_distribution.csv',
    ]

    for fd in ds:
        show_distribution(fd)
