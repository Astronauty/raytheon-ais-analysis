import torch
import gpytorch
from tqdm import tqdm

def train_model_safely(model, likelihood, X_indices, Y_flat, num_epochs=100, lr=0.1):
    """
    Safely train a GP model with robust error handling and numerical stability settings.
    
    Args:
        model: GPyTorch model instance
        likelihood: GPyTorch likelihood instance
        X_indices: Input tensor
        Y_flat: Target tensor (flattened)
        num_epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Trained model, likelihood, and losses
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    losses = []
    
    model.train()
    likelihood.train()
    
    for i in tqdm(range(num_epochs), desc="Training"):
        optimizer.zero_grad()
        
        # Use multiple numerical stability settings
        with gpytorch.settings.cholesky_jitter(1e-3), \
             gpytorch.settings.max_preconditioner_size(15), \
             gpytorch.settings.min_preconditioning_size(2):
            
            try:
                output = model(X_indices)
                loss = -mll(output, Y_flat)
                losses.append(loss.item())
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Loss is {loss.item()}, skipping backward")
                    continue
                    
                loss.backward()
                
                if (i+1) % 10 == 0:
                    print(f'Epoch {i+1}/{num_epochs} - Loss: {loss.item():.4f}')
                    
                optimizer.step()
                
            except Exception as e:
                print(f"Error during training: {str(e)}")
                print(f"Skipping this iteration and continuing...")
                continue
    
    # Set to evaluation mode
    model.eval()
    likelihood.eval()
    
    return model, likelihood, losses
