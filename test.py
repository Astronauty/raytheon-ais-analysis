from gpytorch.lazy import KroneckerProductLazyTensor as KroneckerProductLazyTensor  # Try this if using newer GPyTorch
from gpytorch.lazy import DiagLazyTensor, NonLazyTensor


### Approx GP it code
# num_trajectories = len(gp_regression_dataset)
num_trajectories = 5
models = {}
likelihoods = {}
losses = {}

from multioutput_state_space_gp import ApproximateStateSpaceGPModel

for idx in range(num_trajectories):
    print(f"\nFitting GP for trajectory {idx+1}/{num_trajectories} for MMSI {mmsi}")
    
    # mmsi, times, state_trajectory = gp_regression_dataset[idx]
    mmsi, times, state_trajectory = gp_regression_dataset[idx]
    
    times_tensor = times.detach().to(device)
    
    X_indices = torch.arange(len(times), device=device).unsqueeze(1)
    Y = state_trajectory.detach().to(device) # Targets are shape [time_steps, n_states]
    num_tasks = Y.shape[1]
    
    print(f"X_indices shape: {X_indices.shape}")
    print(f"Y shape: {Y.shape}")
    
    # Define inducing points and indices
    num_inducing = min(5, int(0.1*len(times)))  # Ensure we have roughly 10 percent of data as inducing points or least 5 inducing points
    inducing_indices = X_indices[torch.randperm(len(X_indices))[:num_inducing]]
    inducing_indices = inducing_indices.float()  # Convert to float
    
    print(f"Inducing indices: {inducing_indices}")
    print(f"Inducing indices shape: {inducing_indices.shape}")
    # Create model
    model = ApproximateStateSpaceGPModel(
        inducing_indices, 
        # timesteps=times_tensor, 
        num_tasks=num_tasks
    ).to(device)

    # Use the same likelihood
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.05)

    # Use VariationalELBO instead of ExactMarginalLogLikelihood
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(X_indices))

    # Training loop
    model.train()
    likelihood.train()


    for i in tqdm(range(100), desc=f"Trajectory {idx+1}/{num_trajectories}"):
        optimizer.zero_grad()
        
        try:
            inducing_Y = Y[inducing_indices.squeeze().long()]  # Get the corresponding Y values for inducing points
            output = model(inducing_indices)
            loss = -mll(output, inducing_Y)
            loss.backward()
            optimizer.step()
        except RuntimeError as e:
            print(f"Error: {e}")
            # Print more debugging info
            print(f"inducing_indices shape: {inducing_indices.shape}")
            print(f"inducing_Y shape: {inducing_Y.shape}")
            if 'output' in locals():
                print(f"output mean shape: {output.mean.shape}")

        
        if (i+1) % 10 == 0:
            print(f'MMSI {mmsi} - Epoch {i+1}/100 - Loss: {loss.item():.4f}')












#####
class ApproximateStateSpaceGPModel(ApproximateGP):
    def __init__(self, num_latents, num_tasks=6):
        inducing_points = torch.rand(num_latents, )
        
        # Set up variational distribution (one per inducing point)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-2), batch_shape=torch.Size([num_tasks]))
        
        # Set up variational strategy
        variational_strategy = LMCVariationalStrategy(VariationalStrategy(
            self, inducing_points, variational_distribution, 
            learn_inducing_locations=True),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1                       
            )
        
        super().__init__(variational_strategy)
        
        # Mean module
        # self.mean_module = gpytorch.means.MultitaskMean(
        #     gpytorch.means.ConstantMean(), num_tasks=num_tasks
        # )
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        
        # State space kernel
        self.base_kernel = StateSpaceKernel(batch_shape=torch.Size(batch_shape=[num_latents]))
        
        # Multitask kernel on top of state space kernel
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            self.base_kernel, num_tasks=num_tasks, rank=1
        )
        self.num_tasks = num_tasks
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
