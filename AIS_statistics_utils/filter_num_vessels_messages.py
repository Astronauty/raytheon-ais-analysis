import requests
import pandas as pd
import zipfile
import urllib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path

vessel_groups = {}
vessel_groups["Cargo"] = np.concatenate([np.arange(70, 80), [1003, 1004, 1016]])
vessel_groups["Fishing"] = np.array([30, 1001, 1002])
vessel_groups["Military"] = np.array([35])
vessel_groups["Not Available"] = np.array([0])
vessel_groups["Other"] = np.concatenate([np.arange(1, 21), np.arange(23, 30), np.arange(33, 35), np.arange(38, 52), np.arange(53, 60), np.arange(90, 1000), np.arange(1005, 1012), [1018, 1020, 1022]])
vessel_groups["Passenger"] = np.concatenate([np.arange(60, 70), np.arange(1012, 1016)])
vessel_groups["Pleasure Craft"] = np.array([36, 37, 1019])
vessel_groups["Tanker"] = np.concatenate([np.arange(80, 90), [1017, 1024]])
vessel_groups["Tug Tow"] = np.array([21, 22, 31, 32, 52, 1023, 1025])

map_to_vessel_group = dict()
for vg in vessel_groups:
    for i in vessel_groups[vg]:
        map_to_vessel_group[int(i)] = vg
vessel_group_to_id = dict()
for i, v in enumerate(set(map_to_vessel_group.values())):
    vessel_group_to_id[v] = i

def generate_hourly_stat(df):
    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
    # df = df.sort_values(by='BaseDateTime').reset_index(drop=True)
    df['vessel_group'] = df['VesselType'].apply(lambda x : 'Other' if pd.isna(x) else  map_to_vessel_group[int(x)])
    df['hour'] = df['BaseDateTime'].dt.hour
    
    dfg = df.groupby(['vessel_group', 'hour'])

    n_vessels = dfg['MMSI'].nunique()
    n_messages = dfg['MMSI'].count()

    vessel_df = n_vessels.unstack(level='vessel_group').fillna(0)
    
    messages_df = n_messages.unstack(level='vessel_group').fillna(0)
    return vessel_df, messages_df

def get_timedelta_stat(df, date):
    df = df.sort_values(by='BaseDateTime').reset_index(drop=True)
    df = df.sort_values(by='MMSI').reset_index(drop=True)
    df['vessel_group'] = df['VesselType'].apply(lambda x : 'Other' if pd.isna(x) else  map_to_vessel_group[int(x)])

    df['vessel_group_id'] = df['vessel_group'].apply(lambda s : vessel_group_to_id[s])
    times = np.zeros((9, 86400))
    last_mmsi = 0
    last_time = 0

    for index, row in df.iterrows():
        if index == 0:
            last_mmsi = row['MMSI']
            last_time = row['BaseDateTime']
        else:
            mmsi = row['MMSI']
            time = row['BaseDateTime']
            vg = row['vessel_group_id']
            if mmsi == last_mmsi:
                delta = (time - last_time).seconds
                times[vg, delta] += 1
                last_mmsi =mmsi
                last_time = time
            else:
                last_mmsi = mmsi
                last_time = time 
    np.save(f'npys/delta_{date}.npy', times)

def get_all_stat():
    date_ranges = pd.date_range(start='2023-01-01', end='2023-12-31')
    vdfs = []
    mdfs = []
    save_every = 10
    step = 0
    for date in date_ranges:
        step += 1
        print(f'Date: {date}')
        try: 
            df = pd.read_csv(f'data/AIS_{date.strftime("%Y_%m_%d")}.csv')

        except:
            continue
        vdf, mdf = generate_hourly_stat(df)
        vdf['day'] = date.day
        vdf['month'] = date.month
        vdf['year'] = date.year
        mdf['day'] = date.day
        mdf['month'] = date.month
        mdf['year'] = date.year
        mdfs.append(mdf)
        vdfs.append(vdf)
        if step % save_every == 0:
            mdf_all = pd.concat(mdfs, join='outer')
            vdf_all = pd.concat(vdfs, join='outer')
            mdf_all.to_csv('num_messages.csv')
            vdf_all.to_csv('num_vessels.csv')

if __name__ == '__main__':
    get_all_stat()