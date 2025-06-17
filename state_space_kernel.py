import numpy as np 
import torch
import gpytorch
from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process.kernels import Kernel, RBF, Hyperparameter
from control import *
from scipy.linalg import expm, eigh
import numpy as np
import torch
import gpytorch
from scipy.linalg import expm
import time
from functools import lru_cache
import threading
import torch.nn.functional as F


class StateSpaceKernel(gpytorch.kernels.Kernel):
    def __init__(self, timesteps, m=1E5, I=1E3, q=1.0, r=1.0, dt=1.0):
        """
        State space kernel for a planar ship model.
        Parameters:
        timesteps: tensor of timestamps
        m: mass of the ship
        I: moment of inertia of the ship
        q: process noise covariance
        r: measurement noise covariance
        dt: time step
        """
        # Initialize cache lock BEFORE calling parent constructor
        self.cache_lock = threading.Lock()
        
        # Call parent constructor directly
        gpytorch.kernels.Kernel.__init__(self)
        
        # Cache optimization settings
        self.time_quantization = 0.05  # Quantize time differences to 0.05 units
        self.ku_threshold = 1e-4      # Skip contributions smaller than this
        self.time_diff_threshold = 1e-3  # Threshold for considering timestamps identical
        self.jitter_factor = 1e-2     # Increased jitter factor for numerical stability
        
        # Register parameters with constraints
        self.register_parameter("raw_m", torch.nn.Parameter(torch.tensor(float(np.log(m)))))
        self.register_parameter("raw_I", torch.nn.Parameter(torch.tensor(float(np.log(I)))))

        # Register constraints
        self.register_constraint("raw_m", gpytorch.constraints.Positive())
        self.register_constraint("raw_I", gpytorch.constraints.Positive())

        self.dt = dt
        self.register_buffer("timesteps", timesteps)
        
        # Check if CUDA is available for GPU acceleration
        self.use_gpu = torch.cuda.is_available()
        self._device = timesteps.device
        
        self.n_states = 6
        self.n_inputs = 3
        self.n_outputs = 6
        
        # Force kernels (one per input)
        self.force_kernels = torch.nn.ModuleList()
        for i in range(self.n_inputs):
            # Create RBF kernel with explicit lengthscale initialization
            base_kernel = gpytorch.kernels.RBFKernel()
            # Initialize lengthscale - critical for learning dynamics
            base_kernel.lengthscale = 1.0
            
            # Wrap in ScaleKernel and initialize outputscale
            scaled_kernel = gpytorch.kernels.ScaleKernel(base_kernel)
            scaled_kernel.outputscale = 1.0
            
            self.force_kernels.append(scaled_kernel)
            
        # Cache matrix exponentials for system dynamics
        self._A = None
        self._A_np = None
        self._B = None
        self._C = None
        
        # Optimized cache structure for matrix exponentials
        self.exp_cache = {}
        
        # Initialize system matrices once to avoid recalculating constantly
        self._setup_system_matrices()
        
        # Pre-compute and cache valid timestep combinations
        with torch.no_grad():
            self.timesteps_np = timesteps.cpu().numpy()
            
            # Create all possible time difference pairs in advance
            n_timesteps = len(timesteps)
            self.all_time_diffs = np.zeros((n_timesteps, n_timesteps))
            for i in range(n_timesteps):
                for j in range(n_timesteps):
                    self.all_time_diffs[i,j] = timesteps[i].item() - timesteps[j].item()
            
            # Pre-compute all valid time masks
            self.time_masks = {}
            for t_idx in range(len(timesteps)):
                t_val = timesteps[t_idx].item()
                mask = self.timesteps_np <= t_val
                self.time_masks[t_idx] = mask
                
        # Cache for kernel evaluations (using LRU for automatic management)
        self._kernel_eval = lru_cache(maxsize=1024)(self._compute_kernel_eval)
        
        # JIT compile critical functions if possible
        try:
            if not torch.jit.is_scripting() and not torch.jit.is_tracing():
                self._matrix_mult_impl = torch.jit.script(self._matrix_mult_impl)
                self._matrix_mult_direct = self._matrix_mult_impl
            else:
                self._matrix_mult_direct = self._matrix_mult_impl
        except Exception:
            self._matrix_mult_direct = self._matrix_mult_impl
            
    # Properties for transformed parameters
    @property
    def m(self):
        return self.raw_m_constraint.transform(self.raw_m)
    
    @property
    def I(self):
        return self.raw_I_constraint.transform(self.raw_I)
        
    def _get_device(self):
        """Get current device"""
        return self._device

    def to(self, device=None, dtype=None, non_blocking=False):
        """Override to method to handle device transfers properly"""
        self._device = device if device is not None else self._device
        return super().to(device=device, dtype=dtype, non_blocking=non_blocking)

    def _matrix_mult_impl(self, Ak, Bi_BiT, Al_T, ku):
        return ku * (Ak @ Bi_BiT @ Al_T)
    
    def _matrix_mult(self, Ak, Bi_BiT, Al_T, ku):
        """Safe wrapper around _matrix_mult_impl that handles type conversions"""
        if not isinstance(Al_T, torch.Tensor):
            Al_T = torch.tensor(Al_T, device=self._device)
        if not isinstance(ku, torch.Tensor):
            ku = torch.tensor(ku, device=self._device, dtype=torch.float32)
        
        return self._matrix_mult_direct(Ak, Bi_BiT, Al_T, ku)

    def _setup_system_matrices(self):
        """Create system matrices based on current parameters"""
        m_val = self.m.item()
        I_val = self.I.item()
        
        # Continuous-time system matrices
        A = torch.zeros(6, 6, device=self._device)
        A[0:3, 3:6] = torch.eye(3, device=self._device)
        
        B = torch.zeros(6, 3, device=self._device)
        B[3, 0] = 1/m_val
        B[4, 1] = 1/m_val
        B[5, 2] = 1/I_val
        
        C = torch.eye(6, device=self._device)
        
        # Add small stabilizing terms to diagonal of A for numerical stability
        diag_stabilizer = 1e-6
        A.add_(torch.eye(6, device=self._device) * (-diag_stabilizer))
        
        # Cache the matrices
        self._A = A
        self._A_np = A.cpu().numpy()
        self._B = B
        self._C = C
        
        # Clear exp_cache when parameters change
        with self.cache_lock:
            self.exp_cache = {}
        
        return A, B, C
    
    @lru_cache(maxsize=500)
    def _compute_matrix_exp(self, time_diff_key):
        """Compute matrix exponential with improved numerical stability"""
        time_diff = time_diff_key
        
        # Add damping for numerical stability
        damping = 1e-5
        A_stable = self._A_np.copy()
        
        # Add small damping to diagonal for stability
        for i in range(A_stable.shape[0]):
            A_stable[i, i] -= damping
            
        try:
            exp_result = expm(A_stable * time_diff)
            # Check for NaNs or Infs
            if np.any(np.isnan(exp_result)) or np.any(np.isinf(exp_result)):
                raise ValueError("Matrix exponential produced NaNs or Infs")
            
            return torch.tensor(exp_result, device=self._device, dtype=torch.float32)
        except Exception as e:
            # Fallback with higher damping if regular method fails
            print(f"Matrix exponential failed, using more stable fallback: {e}")
            A_stable = self._A_np.copy()
            for i in range(A_stable.shape[0]):
                A_stable[i, i] -= 0.05  # Higher damping
            exp_result = expm(A_stable * time_diff)
            return torch.tensor(exp_result, device=self._device, dtype=torch.float32)
    
    def _get_matrix_exp(self, time_diff):
        """Get cached matrix exponential or compute and cache it"""
        # Stronger quantization to increase cache hits
        time_diff_quantized = round(time_diff * 20) / 20  # Quantize to 0.05 time units
        
        # Add small epsilon to avoid exactly zero values which can cause issues
        if abs(time_diff_quantized) < 1e-8:
            time_diff_quantized = 1e-8
            
        return self._compute_matrix_exp(time_diff_quantized)
    
    def _compute_kernel_eval(self, kernel_idx, t1_idx, t2_idx):
        """Compute kernel evaluation between two sets of timesteps"""
        kernel = self.force_kernels[kernel_idx]
        
        # Convert to int if tensors
        if hasattr(t1_idx, 'item'):
            t1_idx = t1_idx.item()
        if hasattr(t2_idx, 'item'):
            t2_idx = t2_idx.item()
        
        # Safety check for indices
        if t1_idx not in self.time_masks:
            # Handle missing index - return empty result
            return torch.zeros(1, device=self.timesteps.device), np.array([]), np.array([])
        if t2_idx not in self.time_masks:
            return torch.zeros(1, device=self.timesteps.device), np.array([]), np.array([])
        
        # Get valid timesteps masks
        tk_mask = self.time_masks[t1_idx]
        tl_mask = self.time_masks[t2_idx]
        
        tk_indices = np.where(tk_mask)[0]
        tl_indices = np.where(tl_mask)[0]
        
        # Extract the required timesteps
        tk_values = self.timesteps_np[tk_indices]
        tl_values = self.timesteps_np[tl_indices]
        
        # Convert to tensors for kernel evaluation
        tk_tensor = torch.tensor(tk_values, device=self._device).unsqueeze(1)
        tl_tensor = torch.tensor(tl_values, device=self._device).unsqueeze(1)
        
        # Evaluate the kernel
        with torch.no_grad():
            ku_mat = kernel(tk_tensor, tl_tensor).to_dense()
        
        return ku_mat, tk_indices, tl_indices
    
    def _ensure_psd(self, matrix, min_eigenvalue=1e-5):
        """Ensure a matrix is positive semi-definite by adding jitter to eigenvalues if needed"""
        # Convert to NumPy for eigenvalue computation
        matrix_np = matrix.cpu().numpy()
        
        # Compute eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = eigh(matrix_np)
            
            # Check if any eigenvalues are negative or very close to zero
            min_eig = np.min(eigenvalues)
            if min_eig < min_eigenvalue:
                # Add a small jitter to make it PSD
                jitter = min_eigenvalue - min_eig + 1e-6
                diag_jitter = torch.eye(matrix.shape[0], device=matrix.device) * jitter
                matrix = matrix + diag_jitter
        except Exception as e:
            # If eigenvalue decomposition fails, add jitter directly
            jitter = min_eigenvalue + 1e-6
            diag_jitter = torch.eye(matrix.shape[0], device=matrix.device) * jitter
            matrix = matrix + diag_jitter
            
        return matrix
        
    def _latent_force_cov(self, t1_idx, t2_idx):
        """Highly optimized computation of covariance between states"""
        # Initialize result with zeros using proper memory layout
        cov = torch.zeros(self.n_states, self.n_states, device=self._device, 
                          dtype=torch.float32)
        
        # Get timestamp values
        t1 = self.timesteps[t1_idx].item()
        t2 = self.timesteps[t2_idx].item()
        
        # Skip computation if timestamps are nearly identical (within small tolerance)
        if abs(t1 - t2) < self.time_diff_threshold:
            # For nearly identical timestamps, return a simplified covariance matrix
            for i in range(self.n_inputs):
                Bi = self._B[:, i:i+1]
                Bi_BiT = Bi @ Bi.T
                cov.add_(Bi_BiT)
            
            # Add jitter for numerical stability - higher for diagonal terms
            jitter = cov.max().item() * self.jitter_factor
            cov.add_(torch.eye(self.n_states, device=self._device) * jitter)
            return self._ensure_psd(cov)

        # Compute covariance contributions for each input dimension
        for i in range(self.n_inputs):
            # Get cached kernel evaluations and indices
            ku_mat, tk_indices, tl_indices = self._kernel_eval(i, t1_idx.item(), t2_idx.item())
            
            # Skip if there are no valid indices
            if len(tk_indices) == 0 or len(tl_indices) == 0:
                continue
                
            # Extract input matrix column and pre-compute product
            Bi = self._B[:, i:i+1]
            Bi_BiT = Bi @ Bi.T
            
            # Use larger batch sizes for GPU
            batch_size = 100 if self.use_gpu else 50
            
            # Pre-compute all needed Al matrices
            Al_matrices = {}
            unique_diffs = set()
            for l in tl_indices:
                tl = self.timesteps_np[l]
                diff = t2 - tl
                diff_key = round(diff * 20) / 20  # Stronger quantization
                unique_diffs.add(diff_key)
            
            # Only compute matrix exp for unique time differences
            for diff_key in unique_diffs:
                Al_matrices[diff_key] = self._get_matrix_exp(diff_key)
            
            # Process in batches
            for k_batch_start in range(0, len(tk_indices), batch_size):
                k_batch_end = min(k_batch_start + batch_size, len(tk_indices))
                
                # Pre-compute all Ak matrices for this batch
                Ak_matrices = {}
                for k_idx in range(k_batch_start, k_batch_end):
                    k = tk_indices[k_idx]
                    tk = self.timesteps_np[k]
                    diff = t1 - tk
                    diff_key = round(diff * 20) / 20  # Stronger quantization
                    if diff_key not in Ak_matrices:
                        Ak_matrices[diff_key] = self._get_matrix_exp(diff_key)
                
                # Process each batch item (with vectorization where possible)
                batch_cov = torch.zeros_like(cov)
                for k_pos, k_idx in enumerate(range(k_batch_start, k_batch_end)):
                    k = tk_indices[k_idx]
                    tk = self.timesteps_np[k]
                    diff_key = round((t1 - tk) * 20) / 20
                    Ak = Ak_matrices[diff_key]
                    
                    for l_idx, l in enumerate(tl_indices):
                        tl = self.timesteps_np[l]
                        diff_key = round((t2 - tl) * 20) / 20
                        Al_T = Al_matrices[diff_key].T
                        ku = ku_mat[k_idx - k_batch_start, l_idx].item()
                        
                        # Skip very small contributions
                        if abs(ku) < self.ku_threshold:
                            continue
                        
                        # Calculate contribution
                        if not isinstance(ku, torch.Tensor):
                            ku = torch.tensor(ku, device=self._device)
                            
                        contribution = ku * (Ak @ Bi_BiT @ Al_T)
                        batch_cov.add_(contribution)
                
                # Add batch contribution
                cov.add_(batch_cov)
        
        # Add jitter to ensure PSD - scale based on matrix values
        jitter = cov.max().item() * self.jitter_factor
        jitter = max(jitter, 1e-3)  # Ensure minimum jitter
        
        # Add more to diagonal than off-diagonal elements
        cov.add_(torch.eye(self.n_states, device=self._device) * jitter)
        
        # Make symmetric to ensure numerical stability
        cov = 0.5 * (cov + cov.t())
        
        # Final check and correction for PSD
        return self._ensure_psd(cov)
        
    def _batch_compute_kernel_matrix(self, x1_batch, x2_batch):
        """Compute kernel matrix for a batch of inputs"""
        batch_size1 = x1_batch.size(0)
        batch_size2 = x2_batch.size(0)
        
        # Pre-allocate result tensor
        K_batch = torch.zeros(batch_size1, batch_size2, device=self._device)
        
        # Compute kernel values for the batch
        for i in range(batch_size1):
            for j in range(batch_size2):
                t1_idx, t2_idx = x1_batch[i, 0].long(), x2_batch[j, 0].long()
                cov_x = self._latent_force_cov(t1_idx, t2_idx)
                output_cov = self._C @ cov_x @ self._C.T
                K_batch[i, j] = torch.norm(output_cov, p='fro') ** 2
                
        return K_batch

    def forward(self, x1, x2=None, diag=False, **params):
        """
        Compute the kernel matrix between inputs x1 and x2.
        """
        # Move inputs to the right device
        x1 = x1.to(self._device)
        if x2 is not None:
            x2 = x2.to(self._device)
        else:
            x2 = x1
            
        if diag:
            return self._forward_diag(x1, x2)
            
        n1, n2 = x1.size(0), x2.size(0)
        
        # Make sure system matrices are set up
        if self._A is None or self._B is None or self._C is None:
            self._setup_system_matrices()
        
        # Pre-allocate output tensor
        K = torch.zeros(n1, n2, device=self._device)
        
        # Larger batch sizes for GPU processing
        batch_size = 20 if self.use_gpu else 10
        
        # Process in batches
        for i_start in range(0, n1, batch_size):
            i_end = min(i_start + batch_size, n1)
            x1_batch = x1[i_start:i_end]
            
            for j_start in range(0, n2, batch_size):
                j_end = min(j_start + batch_size, n2)
                x2_batch = x2[j_start:j_end]
                
                # Compute batch of kernel values
                K_batch = self._batch_compute_kernel_matrix(x1_batch, x2_batch)
                K[i_start:i_end, j_start:j_end] = K_batch
        
        # Ensure the matrix is PSD by adding jitter to diagonal
        if x1 is x2:
            # Only add jitter if computing self-covariance
            jitter_base = K.diag().mean() * 0.01  # 1% of mean diagonal value
            jitter = max(jitter_base, 0.1)  # Ensure minimum jitter
            K.add_(torch.eye(n1, device=self._device) * jitter)
            
            # Make symmetric to handle numerical issues
            K = 0.5 * (K + K.t())
            
            # Check eigenvalues and fix if needed (larger minimum eigenvalue)
            K_np = K.cpu().numpy()
            try:
                min_eig = np.linalg.eigvalsh(K_np).min()
                if min_eig < 0.01:  # Ensure all eigenvalues are reasonably positive
                    extra_jitter = 0.01 - min_eig + 0.01
                    K.add_(torch.eye(n1, device=self._device) * extra_jitter)
            except:
                # If eigenvalue computation fails, add more jitter
                K.add_(torch.eye(n1, device=self._device) * 0.1)
        
        # Periodically clear LRU caches
        if self.training and torch.rand(1).item() < 0.01:
            self._compute_matrix_exp.cache_clear()
            self._kernel_eval.cache_clear()
            
        return K

    def _forward_diag(self, x1, x2):
        """Compute diagonal of the kernel matrix more efficiently"""
        n = x1.size(0)
        diag_values = torch.zeros(n, device=self._device)
        
        for i in range(n):
            t1_idx = x1[i, 0].long()
            t2_idx = x2[i, 0].long() if x2 is not x1 else t1_idx
            cov_x = self._latent_force_cov(t1_idx, t2_idx)
            output_cov = self._C @ cov_x @ self._C.T
            diag_values[i] = torch.norm(output_cov, p='fro') ** 2
        
        # Add jitter to diagonal for stability
        diag_values.add_(diag_values.mean() * self.jitter_factor)
        
        return diag_values