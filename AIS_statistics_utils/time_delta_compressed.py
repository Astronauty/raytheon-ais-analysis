import numpy as np

import os
import numpy as np
import pandas as pd


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

vessel_group_id_to_group = dict()
for i in range(len(vessel_group_to_id)):
    for k in vessel_group_to_id:
        if vessel_group_to_id[k] == i:
            vessel_group_id_to_group[i] = k

        

if __name__=='__main__':
    
    slices = [slice(0, 2), slice(2, 5), slice(5,10), slice(10, 30), slice(30,60),slice(60,60*2),slice(60*2, 60*5), slice(60*5,60*10), slice(600, 1800), slice(1800, 3600), 
          slice(3600, 3600*2) ,slice(3600*2, 3600*5), slice(3600*5, 3600*10), slice(3600*10, 3600*15), slice(15*3600, 20*3600), slice(20*3600, 24*3600)]
    slice_name = ['<2 sec','2-5 sec', '5-10 sec', '10-30 sec', '30-60 sec', '1-2 min','2-5 min', '5-10 min', '10-30 min', '30-60 min', '1-2 hour', '2-5 hour', '5-10 hour', '10-15 hour', '15-20 hour', '20-24 hour']
    
    result = []


    data_dict = dict()
    date_ranges = pd.date_range(start='2023-01-01', end='2023-12-31')


    for date in date_ranges:
        d = str(date.date()).replace('-','_')
        # day = date.date()
        # date.npy file contains a numpy array of shape [Num Groups] X 86400 
        #  - j'th row and i'th column contains number of ships of the group j that have delta = i seconds in their consecutive messages.
        if os.path.exists(f'npys/{d}.npy'):
            npf = np.load(f'npys/{d}.npy')
            npf_out = np.zeros((npf.shape[0], len(slices)))
            for i, sl in enumerate(slices):
                npf_out[:, i] = npf[:, sl].sum(axis=1)
            for j in range(npf.shape[0]):
                r = dict(date = str(date.date()), group_id = j, group = vessel_group_id_to_group[j])
                for m in range(len(slices)):
                    r[slice_name[m]] = int(npf_out[j, m])
                result.append(r)
    pd.DataFrame(result).to_csv('time_delta.csv',index=False)

