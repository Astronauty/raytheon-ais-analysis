"""
StateSpaceKernelAdapter module for integrating state space kernels with GPyTorch.
This module provides adapters to make the state space kernel compatible with
GPyTorch's expectations for multi-output Gaussian Processes.
"""
import torch
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.lazy import LazyTensor, NonLazyTensor

class StateSpaceKernelAdapter(gpytorch.kernels.Kernel):
    """
    An adapter kernel that transforms the output of the StateSpaceKernel to be
    compatible with GPyTorch's multi-output GP framework.
    
    The adapter handles the conversion between the [N, N, D, D] state covariance tensor
    from the state space model to the [N*D, N*D] covariance matrix expected by GPyTorch.
    """
    def __init__(self, base_kernel, num_outputs=6, **kwargs):
        """
        Args:
            base_kernel: A StateSpaceKernel instance
            num_outputs: Number of output dimensions (typically 6 for vessel state)
        """
        super(StateSpaceKernelAdapter, self).__init__(**kwargs)
        self.base_kernel = base_kernel
        self.num_outputs = num_outputs
    
    def forward(self, x1, x2=None, diag=False, **params):
        """
        Compute and transform the kernel matrix to match GPyTorch's expectations.
        
        Args:
            x1: Input tensor of shape [batch_size_1, 1] (time indices)
            x2: Input tensor of shape [batch_size_2, 1] (time indices, or None for x1)
            diag: Whether to return only the diagonal
            
        Returns:
            A LazyTensor representing the kernel matrix with shape [batch_size_1 * num_outputs, batch_size_2 * num_outputs]
        """
        if x2 is None:
            x2 = x1
        
        # Get dimensions
        batch_size_1 = x1.size(0)
        batch_size_2 = x2.size(0) if x2 is not None else batch_size_1
        
        # Compute the state space covariance 
        # Using last_dim_is_batch=True to get [batch_size_1, batch_size_2, num_outputs, num_outputs]
        state_space_kernel = self.base_kernel(x1, x2, last_dim_is_batch=True)
        
        # The dimensions are now [batch_size_1, batch_size_2, num_outputs, num_outputs]
        # We need to reshape to [batch_size_1 * num_outputs, batch_size_2 * num_outputs]
        reshaped_kernel = self._reshape_kernel_matrix(state_space_kernel, batch_size_1, batch_size_2)
        
        # Return as a LazyTensor
        return NonLazyTensor(reshaped_kernel)
    
    def _reshape_kernel_matrix(self, kernel_matrix, batch_size_1, batch_size_2):
        """
        Reshape the state space kernel matrix from
        [batch_size_1, batch_size_2, num_outputs, num_outputs] to
        [batch_size_1 * num_outputs, batch_size_2 * num_outputs]
        
        Args:
            kernel_matrix: Tensor of shape [batch_size_1, batch_size_2, num_outputs, num_outputs]
            batch_size_1: Size of the first batch dimension
            batch_size_2: Size of the second batch dimension
        
        Returns:
            Tensor of shape [batch_size_1 * num_outputs, batch_size_2 * num_outputs]
        """
        n_outputs = self.num_outputs
        
        # Initialize the output tensor
        reshaped = torch.zeros(
            batch_size_1 * n_outputs,
            batch_size_2 * n_outputs,
            device=kernel_matrix.device
        )
        
        # Fill in the output tensor block-by-block
        for i in range(batch_size_1):
            for j in range(batch_size_2):
                block = kernel_matrix[i, j]  # [num_outputs, num_outputs]
                
                # Fill the corresponding block in the reshaped matrix
                reshaped[
                    i * n_outputs:(i + 1) * n_outputs,
                    j * n_outputs:(j + 1) * n_outputs
                ] = block
        
        return reshaped


class EfficientStateSpaceKernelAdapter(gpytorch.kernels.Kernel):
    """
    A more efficient adapter that uses advanced tensor operations to reshape
    the state space kernel matrix without explicit loops.
    """
    def __init__(self, base_kernel, num_outputs=6, **kwargs):
        """
        Args:
            base_kernel: A StateSpaceKernel instance
            num_outputs: Number of output dimensions (typically 6 for vessel state)
        """
        super(EfficientStateSpaceKernelAdapter, self).__init__(**kwargs)
        self.base_kernel = base_kernel
        self.num_outputs = num_outputs
    
    def forward(self, x1, x2=None, diag=False, **params):
        """
        Compute and transform the kernel matrix to match GPyTorch's expectations.
        
        Args:
            x1: Input tensor of shape [batch_size_1, 1] (time indices)
            x2: Input tensor of shape [batch_size_2, 1] (time indices, or None for x1)
            diag: Whether to return only the diagonal
            
        Returns:
            A LazyTensor representing the kernel matrix with shape [batch_size_1 * num_outputs, batch_size_2 * num_outputs]
        """
        if x2 is None:
            x2 = x1
        
        # Get dimensions
        batch_size_1 = x1.size(0)
        batch_size_2 = x2.size(0) if x2 is not None else batch_size_1
        
        # Compute the state space covariance 
        # Using last_dim_is_batch=True to get [batch_size_1, batch_size_2, num_outputs, num_outputs]
        state_space_kernel = self.base_kernel(x1, x2, last_dim_is_batch=True)
        
        # Efficiently reshape using einsum and view
        reshaped_kernel = self._efficient_reshape(
            state_space_kernel, batch_size_1, batch_size_2
        )
        
        # Return as a LazyTensor
        return NonLazyTensor(reshaped_kernel)
    
    def _efficient_reshape(self, kernel_matrix, batch_size_1, batch_size_2):
        """
        Efficiently reshape the kernel matrix using tensor operations.
        
        Args:
            kernel_matrix: Tensor of shape [batch_size_1, batch_size_2, num_outputs, num_outputs]
            batch_size_1: Size of first batch dimension
            batch_size_2: Size of second batch dimension
        
        Returns:
            Tensor of shape [batch_size_1 * num_outputs, batch_size_2 * num_outputs]
        """
        n_outputs = self.num_outputs
        
        # Use torch.einsum for the reshaping operation
        # This maps the dimensions as follows:
        # [i, j, d1, d2] -> [i * d1, j * d2]
        
        # First, permute to [i, d1, j, d2]
        permuted = kernel_matrix.permute(0, 2, 1, 3)
        
        # Then reshape to [i*d1, j*d2]
        reshaped = permuted.reshape(batch_size_1 * n_outputs, batch_size_2 * n_outputs)
        
        return reshaped
