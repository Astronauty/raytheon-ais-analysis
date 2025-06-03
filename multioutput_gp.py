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

sns.set_theme(style="whitegrid")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiOutputExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_outputs):
        super(MultiOutputExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.num_outputs = num_outputs

        # Define a mean module for multitask GPs
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_outputs
        )

        # Define a covariance module for multitask GPs
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_outputs, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    

def train_model(model, likelihood, train_x, train_y, num_epochs=500, lr=0.01):
    # Log the training loss to TensorBoard
    log_dir = f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    
    model.to(device)
    likelihood.to(device)
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in tqdm(range(num_epochs), desc="GP Training Progress"):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        writer.add_scalar("Training Loss", loss.item(), i)
        writer.add_scalar("Lengthscale", model.covar_module.data_covar_module.lengthscale.item(), i)
        writer.add_scalar("Noise", model.likelihood.noise.item(), i)
        # writer.add_scalar("Mean", model.mean_module.base_means.constant.item(), i)
    
    writer.flush()
    writer.close()
    # print(f"Training completed. Loss: {loss.item()}")    
    
    return loss, model, likelihood
        
def eval_model(model, likelihood, test_x):
    model.eval()
    likelihood.eval
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        
        return observed_pred

def plot_gp(train_x, train_y, test_x, observed_pred):
    # Define a colormap for each state
        colormap = sns.color_palette("colorblind", 6)

        with torch.no_grad():
            # Initialize plot with 2x3 subplots
            f, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()

            # Labels for the DoFs
            dof_labels = ['x (m)', 'y (m)', r'$\theta$ (rad)', r'$\dot{x}$ (m/s)', r'$\dot{y}$ (m/s)', r'$\dot{\theta}$ (rad/s)']

            # Plot predictive means and confidence bounds for DoFs 1, 2, and 3 in the first row
            for i in range(3):
                ax = axes[0, i]
                # Plot training data as black stars
                ax.scatter(train_x.cpu().numpy().flatten(), train_y.cpu().numpy()[:, i], color=colormap[i], marker='*')
                # Plot predictive means
                ax.plot(test_x.cpu().numpy().flatten(), observed_pred.mean[:, i].cpu().numpy(), color=colormap[i])
                # Plot confidence bounds
                lower_bound = lower[:, i].cpu().numpy()
                upper_bound = upper[:, i].cpu().numpy()
                if i == 0 or i == 1:
                    lower_bound /= 1.1
                    upper_bound *= 1.1
                ax.fill_between(test_x.cpu().numpy().flatten(), lower_bound, upper_bound, color=colormap[i], alpha=0.2)
                ax.set_ylabel(dof_labels[i])

            # Plot predictive means and confidence bounds for DoFs 4, 5, and 6 in the second row
            for i in range(3, 6):
                ax = axes[1, i - 3]
                # Plot training data as black stars
                ax.scatter(train_x.cpu().numpy().flatten(), train_y.cpu().numpy()[:, i], color=colormap[i], marker='*')
                # Plot predictive means
                ax.plot(test_x.cpu().numpy().flatten(), observed_pred.mean[:, i].cpu().numpy(), color=colormap[i])
                # Plot confidence bounds
                lower_bound = lower[:, i].cpu().numpy()
                upper_bound = upper[:, i].cpu().numpy()
                if i == 0 or i == 1:
                    lower_bound /= 1.1
                    upper_bound *= 1.1
                ax.fill_between(test_x.cpu().numpy().flatten(), lower_bound, upper_bound, color=colormap[i], alpha=0.2)
                ax.set_ylabel(dof_labels[i])

            # Set common x-label
            for ax in axes[-1, :]:
                ax.set_xlabel('Time (s)')

            # Create a single legend
            # legend_elements = [
            #     plt.Line2D([0], [0], color=colormap[i], lw=2, label=f'DoF {dof_labels[i]}') for i in range(6)
            # ]
            legend_elements = []
            legend_elements.append(plt.Line2D([0], [0], color='black', marker='*', linestyle='None', label='Observed Data'))
            legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='-', label='Predictive Mean'))
            f.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.05))
            # Increase font size of axis labels
            for ax in axes.flat:
                ax.xaxis.label.set_size(14)
                ax.yaxis.label.set_size(14)
            plt.tight_layout()
            plt.show()
