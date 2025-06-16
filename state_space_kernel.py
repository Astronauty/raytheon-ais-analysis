import numpy as np 
import torch
import gpytorch
from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process.kernels import Kernel, RBF, Hyperparameter
from control import *
from scipy.linalg import expm
import numpy as np
import torch
import gpytorch
from scipy.linalg import expm


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
        super(StateSpaceKernel, self).__init__()
        
        # Register parameters with constraints
        self.register_parameter("raw_m", torch.nn.Parameter(torch.tensor(float(np.log(m)))))
        self.register_parameter("raw_I", torch.nn.Parameter(torch.tensor(float(np.log(I)))))

        # Register constraints
        self.register_constraint("raw_m", gpytorch.constraints.Positive())
        self.register_constraint("raw_I", gpytorch.constraints.Positive())

        self.dt = dt
        self.register_buffer("timesteps", timesteps)
        
        self.n_states = 6
        self.n_inputs = 3
        self.n_outputs = 6
        
        # Force kernels (one per input)
        # self.force_kernels = torch.nn.ModuleList([
        #     gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) 
        #     for _ in range(self.n_inputs)
        # ])
            
        self.force_kernels = torch.nn.ModuleList()
        for i in range(self.n_inputs):
            # Create RBF kernel with explicit lengthscale initialization
            base_kernel = gpytorch.kernels.RBFKernel()
            # Initialize lengthscale - critical for learning dynamics
            base_kernel.lengthscale = 1.0  # Set initial lengthscale
            
            # Wrap in ScaleKernel and initialize outputscale
            scaled_kernel = gpytorch.kernels.ScaleKernel(base_kernel)
            scaled_kernel.outputscale = 1.0  # Set initial outputscale
            
            
            self.force_kernels.append(scaled_kernel)
            
            
        # Cache matrix exponentials for system dynamics
        self._A = None
        self._A_np = None
        
        self.exp_cache = {}
        
        # Print initial kernel parameters to verify
        # self.print_kernel_params()
        
    # Properties for transformed parameters
    @property
    def m(self):
        return self.raw_m_constraint.transform(self.raw_m)
    
    @property
    def I(self):
        return self.raw_I_constraint.transform(self.raw_I)

    def print_kernel_params(self):
        for i, kernel in enumerate(self.force_kernels):
            print(f"Force kernel {i}:")
            print(f"  Outputscale: {kernel.outputscale.item()}")
            print(f"  Lengthscale: {kernel.base_kernel.lengthscale.item()}")
        
    def _setup_system_matrices(self):
        """Create system matrices based on current parameters"""
        m_val = self.m.item()
        I_val = self.I.item()
        # q_val = self.q.item()
        # r_val = self.r.item()
        
        # Continuous-time system matrices
        A = torch.zeros(6, 6, device=self.timesteps.device)
        A[0:3, 3:6] = torch.eye(3, device=self.timesteps.device)
        
        B = torch.zeros(6, 3, device=self.timesteps.device)
        B[3, 0] = 1/m_val
        B[4, 1] = 1/m_val
        B[5, 2] = 1/I_val
        
        C = torch.eye(6, device=self.timesteps.device)
        
        
        # Cache the matrix exponentials for the system dynamics
        self._A = A
        self._A_np = A.cpu().numpy()
        
        
        return A, B, C
    
    # def _latent_force_cov(self, t1_idx, t2_idx):
    #     """Compute covariance between states at t1 and t2"""
    #     A, B, C, Q, R = self._setup_system_matrices()
    #     A_np = A.cpu().numpy()  # For expm calculation
    #     B_np = B.cpu().numpy()
        
    #     cov = torch.zeros(self.n_states, self.n_states, device=self.timesteps.device)
    #     t1 = self.timesteps[t1_idx].item()
    #     t2 = self.timesteps[t2_idx].item()
        
    #     for i in range(self.n_inputs):
    #         kernel = self.force_kernels[i]
            
    #         for k in range(len(self.timesteps)):
    #             tk = self.timesteps[k].item()
    #             if tk > t1:
    #                 continue
                    
    #             # State transition matrix from tk to t1
    #             Ak_np = expm(A_np * (t1 - tk))
    #             Ak = torch.tensor(Ak_np, device=self.timesteps.device)
                
    #             for l in range(len(self.timesteps)):
    #                 tl = self.timesteps[l].item()
    #                 if tl > t2:
    #                     continue
                        
    #                 # State transition matrix from tl to t2
    #                 Al_np = expm(A_np * (t2 - tl))
    #                 Al = torch.tensor(Al_np, device=self.timesteps.device)
                    
    #                 # Get kernel value between times tk and tl
    #                 t1_tensor = torch.tensor([[tk]], device=self.timesteps.device)
    #                 t2_tensor = torch.tensor([[tl]], device=self.timesteps.device)
    #                 # ku = kernel(t1_tensor, t2_tensor).evaluate()

    #                 ku = kernel(t1_tensor, t2_tensor).to_dense()
                    
    #                 # Extract column for current input
    #                 Bi = B[:, i:i+1]
                    
    #                 # Add contribution to covariance
    #                 cov = cov + ku.item() * (Ak @ Bi @ Bi.T @ Al.T)
        
    #     return cov
    
    
    def _get_matrix_exp(self, time_diff):
        """Get cached matrix exponential or compute and cache it"""
        # Round to reduce numerical precision issues in cache keys
        time_diff = round(time_diff, 1)
        
        if time_diff not in self.exp_cache:
            # Compute and cache
            exp_result = expm(self._A_np * time_diff)
            self.exp_cache[time_diff] = torch.tensor(exp_result, device=self.timesteps.device)
        
        return self.exp_cache[time_diff]



    # Vectorized version of _latent_force_cov
    def _latent_force_cov(self, t1_idx, t2_idx):
        A, B, C = self._setup_system_matrices()
        A_np = A.cpu().numpy()
        B_np = B.cpu().numpy()
        cov = torch.zeros(self.n_states, self.n_states, device=self.timesteps.device)
        t1 = self.timesteps[t1_idx].item()
        t2 = self.timesteps[t2_idx].item()

        timesteps = self.timesteps.cpu().numpy()
        tk_valid = timesteps[timesteps <= t1]
        tl_valid = timesteps[timesteps <= t2]

        for i in range(self.n_inputs):
            kernel = self.force_kernels[i]
            Bi = B[:, i:i+1]

            # Vectorized matrix exponentials
            Ak_all = np.stack([expm(A_np * (t1 - tk)) for tk in tk_valid])  # [K, n_states, n_states]
            Al_all = np.stack([expm(A_np * (t2 - tl)) for tl in tl_valid])  # [L, n_states, n_states]

            Ak_all = torch.tensor(Ak_all, device=self.timesteps.device)
            Al_all = torch.tensor(Al_all, device=self.timesteps.device)

            # Vectorized kernel computation
            tk_tensor = torch.tensor(tk_valid, device=self.timesteps.device).unsqueeze(1)
            tl_tensor = torch.tensor(tl_valid, device=self.timesteps.device).unsqueeze(1)
            ku_mat = kernel(tk_tensor, tl_tensor).to_dense()  # [K, L]

            # Batch matrix multiplication
            # cov += sum_{k,l} ku_mat[k,l] * (Ak_all[k] @ Bi @ Bi.T @ Al_all[l].T)
            for k in range(len(tk_valid)):
                Ak = Ak_all[k]
                for l in range(len(tl_valid)):
                    Al = Al_all[l]
                    ku = ku_mat[k, l].item()
                    cov = cov + ku * (Ak @ Bi @ Bi.T @ Al.T)

        return cov
        
    
    ### Use cached matrix exponentials for efficiency
    # def _latent_force_cov(self, t1_idx, t2_idx):
    #     A, B, C = self._setup_system_matrices()
    #     cov = torch.zeros(self.n_states, self.n_states, device=self.timesteps.device)
    #     t1 = self.timesteps[t1_idx].item()
    #     t2 = self.timesteps[t2_idx].item()

    #     timesteps = self.timesteps.cpu().numpy()
    #     tk_valid = timesteps[timesteps <= t1]
    #     tl_valid = timesteps[timesteps <= t2]

    #     for i in range(self.n_inputs):
    #         kernel = self.force_kernels[i]
    #         Bi = B[:, i:i+1]

    #         # Use cached matrix exponentials instead of recomputing
    #         Ak_all = []
    #         for tk in tk_valid:
    #             Ak_all.append(self._get_matrix_exp(t1 - tk))
            
    #         Al_all = []
    #         for tl in tl_valid:
    #             Al_all.append(self._get_matrix_exp(t2 - tl))

    #         # Stack if not already tensors
    #         if not isinstance(Ak_all[0], torch.Tensor):
    #             Ak_all = torch.stack(Ak_all)
    #             Al_all = torch.stack(Al_all)

    #         # Vectorized kernel computation
    #         tk_tensor = torch.tensor(tk_valid, device=self.timesteps.device).unsqueeze(1)
    #         tl_tensor = torch.tensor(tl_valid, device=self.timesteps.device).unsqueeze(1)
    #         ku_mat = kernel(tk_tensor, tl_tensor).to_dense()  # [K, L]

    #         # Compute contributions
    #         for k in range(len(tk_valid)):
    #             Ak = Ak_all[k]
    #             for l in range(len(tl_valid)):
    #                 Al = Al_all[l]
    #                 ku = ku_mat[k, l].item()
    #                 cov = cov + ku * (Ak @ Bi @ Bi.T @ Al.T)

    #     return cov
   
    def forward(self, x1, x2=None, diag=False, **params):
        """
        Compute the kernel matrix between inputs x1 and x2.
        """
        if x2 is None:
            x2 = x1
            
        if diag and not self.training:
            return self._forward_diag(x1, x2)
            
        n1, n2 = x1.size(0), x2.size(0)
        
        _, _, C = self._setup_system_matrices()
        
        # First calculate the full covariance matrix with shape [T, T, D, D]
        # K = torch.zeros(n1, n2, self.n_outputs, self.n_outputs, device=x1.device)
        K = torch.zeros(n1, n2, device=x1.device)
        
        for i in range(n1): 
            for j in range(n2):
                t1_idx, t2_idx = x1[i, 0].long(), x2[j, 0].long()
                cov_x = self._latent_force_cov(t1_idx, t2_idx)
                output_cov = C @ cov_x @ C.T 
                # covar_matrix[i, j, :, :] = output_cov
             
                K[i, j] = torch.norm(output_cov, p='fro') ** 2 
                # K[i, j, :, :] = output_cov
        return K