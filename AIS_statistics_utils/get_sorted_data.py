import pandas as pd
import numpy as np
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

if __name__ == '__main__':
    date_ranges = pd.date_range(start='2023-04-26', end='2023-12-31')
    vdfs = []
    mdfs = []
    step = 0
    for date in date_ranges:
        step += 1
        print(f'Date: {date}')
        try: 
            df = pd.read_csv(f'data/AIS_{date.strftime("%Y_%m_%d")}.csv')
            df = df.sort_values(by='BaseDateTime').reset_index(drop=True)
            df = df.sort_values(by='MMSI').reset_index(drop=True)
            df['vessel_group'] = df['VesselType'].apply(lambda x : 'Other' if pd.isna(x) else  map_to_vessel_group[int(x)])
            df['vessel_group_id'] = df['vessel_group'].apply(lambda s : vessel_group_to_id[s])
            df_ = df[['MMSI', 'BaseDateTime', 'vessel_group_id', 'vessel_group']]
            df_.to_csv(f'sorted_data/AIS_{date.strftime("%Y_%m_%d")}.csv')
        except:
            continue
            
        
        if step % 10 == 0:
            print(step)