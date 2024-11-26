import os
import pandas as pd
import pathlib
import numpy as np
import torch 
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter

class Sing

class AISSingleStepPredictionDataset(Dataset):
    def __init__(self, csv_path ,train): # TODO: handle multiple csv imports
        self.KNOTS_TO_METERS_PER_SECOND = 0.514444
        
        self.df = pd.read_csv(csv_path)

        train_df, test_df = train_test_split(self.df, test_size=0.2)
        
    def __len__(self):
        return self.df.shape[0]-1 # subtract 1 since this is single step state prediction
    
    def __getitem__(self, idx):
        dt = self.df.iloc[idx+1] - [idx]
        
        # Get state at timestep k
        x_k = self.df.iloc[idx]['LON']
        y_k = self.df.iloc[idx]['LAT']
        phi_k = self.df.iloc[idx]['HEADING']
        COG = self.df.iloc[idx]['COG'] # angle that the instantaneous velocity makes in world coords
        SOG = self.df.iloc[idx]['SOG'] # speed of a vessel in relation to a fixed point on the Earth's surface
        xd_k = np.cos(COG) * SOG * self.KNOTS_TO_METERS_PER_SECOND
        yd_k = np.sin(COG) * SOG * self.KNOTS_TO_METERS_PER_SECOND
        phid_k = 0 # TODO: how to better estimate angular rate?
        
        # Get state at timestep kp1
        x_kp1 = self.df.iloc[idx]['LON']
        y_kp1 = self.df.iloc[idx]['LAT']
        phi_kp1 = self.df.iloc[idx]['HEADING']
        COG_kp1 = self.df.iloc[idx]['COG'] # angle that the instantaneous velocity makes in world coords
        SOG_kp1 = self.df.iloc[idx]['SOG'] # speed of a vessel in relation to a fixed point on the Earth's surface
        xd_kp1 = np.cos(COG_kp1) * SOG_kp1 * self.KNOTS_TO_METERS_PER_SECOND
        yd_kp1 = np.sin(COG_kp1) * SOG_kp1 * self.KNOTS_TO_METERS_PER_SECOND
        phid_kp1 = 0 # TODO: how to better estimate angular rate?
        
        # x = [x, y, phi, xd, yd, phid]
        state_k = np.array([x_k, y_k, phi_k, xd_k, yd_k, phid_k])
        state_kp1 = np.array(x_kp1, y_kp1, phi_kp1, xd_kp1, yd_kp1, phid_kp1)
        
        
        return state_k, state_kp1