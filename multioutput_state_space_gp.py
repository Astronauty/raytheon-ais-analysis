import math
import torch
import gpytorch
import seaborn as sns
import numpy as np
from datetime import datetime
from control import *
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from state_space_kernel import * 
from gpytorch.means import MultitaskMean
from gpytorch.kernels import *
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import LMCVariationalStrategy

sns.set_theme(style="whitegrid")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class StateSpaceGPModel(gpytorch.models.ExactGP): 
    def __init__(self, train_x, train_y, likelihood, timesteps, num_tasks=6):
        # self.mean_module = gpytorch.means.ConstantMean()
        self.train_y_shape = train_y.shape
        train_y_reshaped = train_y.reshape(-1)
        
        super().__init__(train_x, train_y_reshaped, likelihood)
        self.num_tasks = num_tasks
        
        self.mean_module = MultitaskMean(
            gpytorch.means.ConstantMean(),
            num_tasks=num_tasks,  # Number of output dimensions
        )

        self.base_kernel = StateSpaceKernel(timesteps) 
        
        self.covar_module = MultitaskKernel(
            self.base_kernel,
            num_tasks=num_tasks,  # Number of output dimensions
            rank=1
        )
        
        # self.covar_module = gpytorch.kernels.MultitaskKernel(
        #     gpytorch.kernels.RBFKernel() + gpytorch.kernels.LinearKernel(),
        #     num_tasks=num_tasks, rank=1
        #     )
        
        self.num_outputs = num_tasks

        
    def forward(self, x):
        mean_x = self.mean_module(x)  # [N]
        covar_x = self.covar_module(x)
        
        # print(f"mean_x shape = {mean_x.shape}, covar_x batch shape = {covar_x.batch_shape}")

        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    

def train_model(model, likelihood, train_x, train_y, num_epochs=500, lr=0.01, mmsi=None, session_id=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Log the training loss to TensorBoard
    if mmsi is None:
        log_dir = f"logs/gp_regression/{session_id}/mmsi_{mmsi}"
    else:
        log_dir = f"logs/gp_regression/{session_id}/model_{datetime.now().strftime('%H-%M-%S')}"

        
    writer = SummaryWriter(log_dir=log_dir)
    
    model.to(device)
    likelihood.to(device)
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in tqdm(range(num_epochs), desc=f"GP Training Progress"):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        tag_prefix = f"{mmsi}/" if mmsi is not None else ""
        writer.add_scalar(f"Training Loss/{tag_prefix}", loss.item(), i)
        # writer.add_scalar(f'Length Scale/{tag_prefix}', model.covar_module.data_covar_module.kernels[0].lengthscale.item(), i)
        # writer.add_scalar(f'Variance/{tag_prefix}', model.covar_module.data_covar_module.kernels[1].variance.item(), i)
        
            
    writer.flush()
    writer.close()
    # print(f"Training completed. Loss: {loss.item()}")    
    return loss, model, likelihood
    
# def train_model(mmsi, times, state_trajectory, num_epochs=100, lr=0.01):
    # times_tensor = times.detach().to(device)
    # X_indices = torch.arange(len(times), device=device).unsqueeze(1)
    # Y = state_trajectory.detach().to(device)
    
    # # likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=6).to(device)
    # likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    # model = StateSpaceGPModel(X_indices, Y, likelihood, timesteps=times_tensor).to(device)
    
    # model.train()
    # likelihood.train()

    # optimizer = torch.optim.Adam([
    #     {'params': model.parameters()},
    # ], lr=lr)
    
    # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # losses = []
    
    # for i in range(num_epochs):
    #     optimizer.zero_grad()
    #     output = model(X_indices)
    #     loss = -mll(output, Y)
    #     losses.append(loss.item())
    #     loss.backward()
        
    #     if (i+1) % 10 == 0:
    #         print(f'Epoch {i+1}/{num_epochs} - Loss: {loss.item():.4f}')
    #     optimizer.step()
    
    # # Set to evaluation mode
    # model.eval()
    # likelihood.eval()
    
    # return model, likelihood, losses

# def eval_model(model, likelihood, times, num_points=200):
    # Create evenly spaced test points
    # test_times = torch.linspace(times.min(), times.max(), num_points, device=device)
    
    # # Convert to indices
    # test_indices = torch.zeros(num_points, 1, device=device)
    # for i, t in enumerate(test_times):
    #     # Find closest time in the original timesteps
    #     closest_idx = torch.argmin(torch.abs(times - t))
    #     test_indices[i, 0] = closest_idx
    
    # # Get predictions
    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #     predictions = likelihood(model(test_indices))
    #     mean = predictions.mean
    #     lower, upper = predictions.confidence_region()
    
    # return test_times, mean, lower, upper

def eval_model(model, likelihood, test_X):
    """Directly return GPyTorch prediction object"""
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-3):
        # This returns a MultitaskMultivariateNormal object
        return likelihood(model(test_X))

