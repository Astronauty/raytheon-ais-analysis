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


sns.set_theme(style="whitegrid")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StateSpaceGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, timesteps):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = StateSpaceKernel(timesteps)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

class StateSpaceMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, timesteps=None):
        """
        Args:
            train_x: Time indices [time_steps, 1]
            train_y: Target values [time_steps, num_outputs] (not flattened!)
            likelihood: MultitaskGaussianLikelihood
            timesteps: Actual time values
        """
        super().__init__(train_x, train_y, likelihood)
        self.num_tasks = train_y.shape[1] 
        
        # Define mean for multitask output
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), 
            num_tasks=self.num_tasks
        )
        
        # Define kernel structure
        # 1. Base kernel for time correlation - this is your state space kernel
        self.time_kernel = StateSpaceKernel(timesteps)
        
        # 2. Task kernel structure - this models correlations between state variables
        # Option A: Independent outputs (diagonal task covariance)
        self.task_covar_module = gpytorch.kernels.IndexKernel(
            num_tasks=self.num_tasks, 
            rank=1  # Can increase for more correlation
        )
        
        # Create multitask kernel structure
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            self.time_kernel, 
            num_tasks=self.num_tasks,
            rank=1  # Can increase for more correlation
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
class MultiOutputStateSpaceGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, timesteps, num_outputs=6):
        """
        A model that handles multi-output state space prediction.
        
        Args:
            train_x: Indices tensor [time_steps, 1]
            train_y: Flattened targets [time_steps * num_outputs]
            likelihood: GPyTorch likelihood
            timesteps: Actual time values
            num_outputs: Number of state dimensions (default 6)
        """
        super().__init__(train_x, train_y, likelihood)
        self.num_outputs = num_outputs
        self.time_steps = train_x.shape[0]
        
        # Register time steps buffer
        if timesteps is not None:
            self.register_buffer("timesteps", timesteps)
        
        # Independent mean for each state variable
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Use state space kernel for covariance
        self.covar_module = StateSpaceKernel(timesteps)
        
        # This handles reshaping between GP and multi-output structure
        self.last_output_shape = None

    def custom_block_diag(*matrices):
        """Custom implementation of block_diag for when torch.block_diag isn't available"""
        device = matrices[0].device if matrices else torch.device('cpu')
        if len(matrices) == 0:
            return torch.zeros(0, 0, device=device)
        if len(matrices) == 1:
            return matrices[0]
        
        # Get matrix shapes
        matrix_shapes = [m.shape for m in matrices]
        output_rows = sum(shape[0] for shape in matrix_shapes)
        output_cols = sum(shape[1] for shape in matrix_shapes)
        
        # Create output tensor
        output = torch.zeros(output_rows, output_cols, device=device)
        
        # Fill diagonal blocks
        row_idx, col_idx = 0, 0
        for matrix in matrices:
            rows, cols = matrix.shape
            output[row_idx:row_idx + rows, col_idx:col_idx + cols] = matrix
            row_idx += rows
            col_idx += cols
        
        return output
        
    def forward(self, x):
        """
        Forward pass that handles reshaping between time steps and state dimensions.
        Optimized for GPyTorch 1.14 based on testing results.
        """
        mean_x = self.mean_module(x)
        
        # Number of time points and output dimensions
        T = x.shape[0]  # Number of time points
        D = self.num_outputs  # Number of output dimensions
        
        # Get the base time covariance from the state space kernel
        try:
            base_covar = self.covar_module(x)
            
            # Check if base_covar is a lazy tensor by checking for evaluate method
            if hasattr(base_covar, 'evaluate'):
                base_covar = base_covar.evaluate()
                
            # Check if we already have a multi-output covariance
            if base_covar.shape[0] == T*D and base_covar.shape[1] == T*D:
                # Already in the right shape, use as is
                dense_covar = base_covar
            else:
                # We need to create a block diagonal structure
                # Based on your test results, torch.block_diag works!
                blocks = [base_covar for _ in range(D)]
                try:
                    dense_covar = torch.block_diag(*blocks)
                except Exception as e:
                    print(f"torch.block_diag failed: {e}, using custom implementation")
                    dense_covar = self.custom_block_diag(*blocks)
        except Exception as e:
            print(f"Error creating covariance: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a simple fallback covariance
            eye_covar = torch.eye(T, device=x.device)
            blocks = [eye_covar for _ in range(D)]
            try:
                dense_covar = torch.block_diag(*blocks)
            except Exception:
                dense_covar = self.custom_block_diag(*blocks)
        
        # Make sure covariance is symmetric and stable
        dense_covar = (dense_covar + dense_covar.transpose(-1, -2)) / 2
        dense_covar = dense_covar + torch.eye(T*D, device=dense_covar.device) * 1e-6
        
        # Repeat mean for each output
        mean_x = mean_x.repeat(D)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, dense_covar)

    def custom_block_diag(self, *matrices):
        """
        Custom implementation of block_diag that works with testing.
        For GPyTorch 1.14 compatibility.
        """
        device = matrices[0].device
        rows = sum(m.shape[0] for m in matrices)
        cols = sum(m.shape[1] for m in matrices)
        result = torch.zeros(rows, cols, device=device)
        r, c = 0, 0
        for m in matrices:
            h, w = m.shape
            result[r:r+h, c:c+w] = m
            r += h
            c += w
        return result
def train_model(mmsi, times, state_trajectory, num_epochs=100, lr=0.01):
    times_tensor = times.detach().to(device)
    X_indices = torch.arange(len(times), device=device).unsqueeze(1)
    Y = state_trajectory.detach().to(device)
    
    # likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=6).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = StateSpaceGPModel(X_indices, Y, likelihood, timesteps=times_tensor).to(device)
    
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=lr)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    
    for i in range(num_epochs):
        optimizer.zero_grad()
        output = model(X_indices)
        loss = -mll(output, Y)
        losses.append(loss.item())
        loss.backward()
        
        if (i+1) % 10 == 0:
            print(f'Epoch {i+1}/{num_epochs} - Loss: {loss.item():.4f}')
        optimizer.step()
    
    # Set to evaluation mode
    model.eval()
    likelihood.eval()
    
    return model, likelihood, losses

def predict_with_model(model, likelihood, times, num_points=200):
    # Create evenly spaced test points
    test_times = torch.linspace(times.min(), times.max(), num_points, device=device)
    
    # Convert to indices
    test_indices = torch.zeros(num_points, 1, device=device)
    for i, t in enumerate(test_times):
        # Find closest time in the original timesteps
        closest_idx = torch.argmin(torch.abs(times - t))
        test_indices[i, 0] = closest_idx
    
    # Get predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(test_indices))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
    
    return test_times, mean, lower, upper

