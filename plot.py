from main import EnergyPlusEnv
import os
os.environ["ENERGYPLUS"] = '/usr/local/EnergyPlus-8-8-0/energyplus'
os.environ["ENERGYPLUS_MODEL"] = '/home/louis/Documents/ResidentialHeatingRL/eplus_models//SimpleResidential.idf'
os.environ["ENERGYPLUS_WEATHER"] = '/usr/local/EnergyPlus-8-8-0/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw'
env = EnergyPlusEnv()
env.plot(csv_file='/tmp/openai-2019-03-09-14-26-47-654019/output/episode-00000000/eplusout.csv.gz')