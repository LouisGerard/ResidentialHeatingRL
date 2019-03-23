if __name__ == '__main__':
    from main import EnergyPlusEnv
    import os
    os.environ["ENERGYPLUS"] = '/usr/local/EnergyPlus-8-8-0/energyplus'
    os.environ["ENERGYPLUS_MODEL"] = '/home/louis/Documents/ResidentialHeatingRL/eplus_models//SimpleResidential.idf'
    os.environ["ENERGYPLUS_WEATHER"] = '/usr/local/EnergyPlus-8-8-0/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw'
    env = EnergyPlusEnv()

    env.plot(csv_file='/tmp/openai-2019-03-16-18-11-03-068261/output/episode-00000009/eplusout.csv.gz')
    print(env.ep_model.show_diff())
