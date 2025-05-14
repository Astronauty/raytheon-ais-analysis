import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

def generate_hourly_plot(figsize=(16, 6)):
    num_messages = pd.read_csv(f'csvs/num_messages.csv')
    num_vessels = pd.read_csv('csvs/num_vessels.csv')
    num_messages['hour'] = num_messages['hour'].astype(int)
    num_vessels['hour'] = num_vessels['hour'].astype(int)

    cols = [x for x in num_messages.columns.tolist() if x not in ['month', 'day', 'hour']]
    for col in cols:
        if col == 'Not Available':
            continue
        nm = num_messages[['hour',col]].groupby('hour').sum().reset_index()
        fig, ax= plt.subplots(figsize=figsize)
        ax2 = ax.twinx()
        ax.plot(nm['hour'], nm[col], label='# Messages', color='red')

        nv = num_vessels[['hour',col]].groupby('hour').sum().reset_index()
        ax2.plot(nv['hour'], nv[col], color='blue', label='# Vessels')
        ax.set_xticks(ticks=range(24), labels=range(24)) 
        ax.set_ylabel('Number of Messages', fontsize=14)
        ax2.set_ylabel('Number of Vessels', fontsize=14)    
        plt.title(f'Number of Messages and Vessels per hour for {col}', fontsize=16)
        ax.legend(fontsize=14, loc='upper left')
        ax.set_ylim(bottom=0, top=nm[col].max() * 1.2)
        ax2.set_ylim(bottom=0, top=nv[col].max() * 1.2)
        # ax.set_yscale('log')
        # ax2.set_yscale('log')
        ax2.legend(fontsize=14,loc = 'upper right')
        plt.savefig(f'plots/hourly/{col}.pdf')
        plt.tight_layout(pad=2)
        plt.savefig(f'plots/hourly{col}.png', dpi=300)
        plt.close()
def generate_daily_plot(figsize=(16, 6)):
    num_messages = pd.read_csv(f'csvs/num_messages.csv')
    num_vessels = pd.read_csv('csvs/num_vessels.csv')
    num_messages['day_month'] = num_messages['day'].astype(str) + '-' + num_messages['month'].astype(str) 
    num_vessels['day_month'] = num_vessels['day'].astype(str) + '-' + num_vessels['month'].astype(str)
    # num_messages = num_messages.sort_values(by=['month', 'day'])
    
    cols = [x for x in num_messages.columns.tolist() if x not in ['month', 'day', 'day_month', 'hour']]
    for col in cols:
        if col == 'Not Available':
            continue
        nm = num_messages[['day_month',col]].groupby('day_month').sum().reset_index()
        nm = nm.sort_values(by=['day_month'], key=lambda x: pd.to_datetime(x, format='%d-%m'))
        # print(nm)
        # import code; code.interact(local=dict(globals(), **locals()))
        fig, ax= plt.subplots(figsize=figsize)
        # plt.figure(figsize=figsize)
        ax2 = ax.twinx()
        ax.plot(nm['day_month'], nm[col], label='# Messages', color='red')

        nv = num_vessels[['day_month',col]].groupby('day_month').sum().reset_index()
        nv = nv.sort_values(by=['day_month'], key=lambda x: pd.to_datetime(x, format='%d-%m'))
        # print(nv)
        # import code; code.interact(local=dict(globals(), **locals()))
        ax2.plot(nv['day_month'], nv[col], color='blue', label='# Vessels')
        # plt.xticks(rotation=90)
        ax.set_xticks(ticks=range(len(num_messages['day_month'].unique()))[::20], labels=nm['day_month'][::20]) 
        ax.set_ylabel('Number of Messages', fontsize=14)
        ax2.set_ylabel('Number of Vessels', fontsize=14)    
        ax.set_ylim(bottom=0, top=nm[col].max() * 1.2)
        ax2.set_ylim(bottom=0, top=nv[col].max() * 1.2)
        ax.legend(fontsize=14, loc='upper left')
        ax2.legend(fontsize=14,loc = 'upper right')
        plt.title(f'Number of Messages and Vessels per day for {col}', fontsize=16)
        plt.savefig(f'plots/daily/{col}.pdf')
        plt.tight_layout(pad=2)
        plt.savefig(f'plots/daily/{col}.png', dpi=300)
        
        plt.close()

def generate_daily_plot_common_num_messages(figsize=(16, 6)):
    num_messages = pd.read_csv(f'csvs/num_messages.csv')
    num_messages = num_messages.drop(columns=['year', 'Not Available', 'Other'])

    num_messages['day_month'] = num_messages['day'].astype(str) + '-' + num_messages['month'].astype(str) 
    # num_vessels['day_month'] = num_vessels['day'].astype(str) + '-' + num_vessels['month'].astype(str)
    # num_messages = num_messages.sort_values(by=['month', 'day'])
    
    cols = [x for x in num_messages.columns.tolist() if x not in ['month', 'day', 'day_month', 'hour', 'year']]
    
    slices = [slice(0, 3), slice(3, 5), slice(5, 7)]
    for i, slc in enumerate(slices):
        fig, ax= plt.subplots(figsize=figsize)
        for col in cols[slc]:
   
            nm = num_messages[['day_month',col]].groupby('day_month').sum().reset_index()
            nm = nm.sort_values(by=['day_month'], key=lambda x: pd.to_datetime(x, format='%d-%m'))
            # print(col)
            # import code; code.interact(local=dict(globals(), **locals()))
            
            # nm_col = nm[col].values/nm[col].values.max()
            nm_col = gaussian_filter1d(nm[col]/nm[col].values.max(), sigma=2)
            ax.plot(nm['day_month'], nm_col, label=col)

            ax.set_xticks(ticks=range(len(num_messages['day_month'].unique()))[::20], labels=nm['day_month'][::20]) 
            # ax.set_ylabel(' Messages', fontsize=14)
            # ax2.set_ylabel('Number of Vessels', fontsize=14)    
        plt.title(f'Messages per day', fontsize=16)
        plt.legend(fontsize=14)
        plt.savefig(f'plots/messages_{i}.pdf')
        plt.tight_layout(pad=2)
        plt.savefig(f'plots/messages_{i}.png', dpi=300)
        plt.close()

def generate_daily_plot_common_num_vessels(figsize=(16, 6)):
    num_vessels = pd.read_csv('csvs/num_vessels.csv')
    num_vessels = num_vessels.drop(columns=['year', 'Not Available', 'Other'])
    num_vessels['day_month'] = num_vessels['day'].astype(str) + '-' + num_vessels['month'].astype(str)
    # num_messages = num_messages.sort_values(by=['month', 'day'])
    
    cols = [x for x in num_vessels.columns.tolist() if x not in ['month', 'day', 'day_month', 'hour', 'year']]
    # fig, ax= plt.subplots(figsize=figsize)
    slices = [slice(0, 3), slice(3, 5), slice(5, 7)]
    for i, slc in enumerate(slices):
        fig, ax= plt.subplots(figsize=figsize)
        for col in cols[slc]:

            # print(nm)
            # import code; code.interact(local=dict(globals(), **locals()))
            
            # plt.figure(figsize=(16, 6))

            nv = num_vessels[['day_month',col]].groupby('day_month').sum().reset_index()
            nv = nv.sort_values(by=['day_month'], key=lambda x: pd.to_datetime(x, format='%d-%m'))
            # print(nv)
            # nv_col = nv[col].values/nv[col].values.max()
            nv_col = gaussian_filter1d(nv[col].values/nv[col].values.max(), sigma=2)
            ax.plot(nv['day_month'], nv_col, label=col)
            # plt.xticks(rotation=90)
            ax.set_xticks(ticks=range(len(num_vessels['day_month'].unique()))[::20], labels=nv['day_month'][::20]) 
            # ax.set_ylabel('Vessels', fontsize=14)
        plt.title(f'Vessels per day', fontsize=16)
        plt.legend(fontsize=14)
        plt.savefig(f'plots/vessels_{i}.pdf')
        plt.tight_layout(pad=2)
        plt.savefig(f'plots/vessels_{i}.png', dpi=300)
        plt.close()

def generate_time_delta_plot():
    time_delta = pd.read_csv('csvs/time_delta.csv')
    time_delta = time_delta.drop(columns=['date', 'group_id'])
    time_delta = time_delta.groupby(['group']).sum().reset_index()
    groups = time_delta['group'].unique()
    plt.figure(figsize=(9, 6))
    for group in groups:
        if group == 'Not Available':
            continue
        plt.plot(time_delta.columns[1:], time_delta[time_delta['group'] == group].values[0][1:], label=group)
        
        # plt.plot(td['day_month'], td[col])
    # plt.xticks(rotation=90)
        # plt.ylabel('Time Delta (s)', fontsize=14)
        # plt.title(f'Messages with Time Delta for {col}', fontsize=16)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout(pad=5)
    plt.yscale('log')
    plt.ylabel('Number of messages', fontsize=12)
    plt.xlabel('Time Delta', fontsize=12)
    # plt.title('Time Delta for each group')
    plt.savefig(f'plots/time_delta.pdf')
    plt.tight_layout(pad=2)
    plt.savefig(f'plots/time_delta.png', dpi=300)
    plt.close()
generate_daily_plot((8,6))
generate_daily_plot_common_num_messages((8,6))
generate_daily_plot_common_num_vessels((8,6))
# generate_time_delta_plot((8,6))
generate_hourly_plot((8,6))

