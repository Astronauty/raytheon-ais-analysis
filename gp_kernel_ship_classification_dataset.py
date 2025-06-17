import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class GPKernelShipClassificationDataset(Dataset):
    """
    Dataset for ship classification using Gaussian Process kernel parameters. Takes fitted GP models and extracts kernel parameters for each ship (identified by MMSI).
    These kernel parameters are used for classification tasks, where each ship is classified into a group based on its kernel parameters.
    Note that the group_id is distinct from the AIS vessel code. The map can be found in the gp_regression_dataset or at https://coast.noaa.gov/data/marinecadastre/ais/VesselTypeCodes2018.pdf
    The dataset can also handle unscaled kernel parameters if scalers are provided for each MMSI.
    """
    def __init__(self, gp_regression_dataset, models, device, scalers_by_mmsi=None):
        """_summary_

        Args:
            gp_regression_dataset (_type_): Dataset containing GP regression data. (Used to extract MMSI and class labels)
            models (_type_): Dictionary of fitted GP models per MMSI, where keys are MMSI and values are GP models.
            device (_type_): _description_
        """
        self.gp_regression_dataset = gp_regression_dataset
        self.models = models
        self.device = device
        self.scalers_by_mmsi = scalers_by_mmsi


        # self.kernel_params = kernel_params # Dictionary of fitted GP kernel parameters per MMSI
        self.data = [] # List of (mmsi, kernel_params_tensor, group_id) tuples
        self.mmsis = [] # List of MMSI identifiers

        gp_kernel_ship_classification = {}
        
        for mmsi in models:
            model = models[mmsi]
            kernel_params = self.extract_kernel_params(model)
            group_id = gp_regression_dataset.get_vessel_group_id_by_mmsi(mmsi)
            gp_kernel_ship_classification[mmsi] = {
                'kernel_params': kernel_params,
                'group_id': group_id,
            }
            
            kernel_params_tensor = torch.tensor(list(kernel_params.values()), dtype=torch.float32)
            self.data.append((mmsi, kernel_params_tensor, group_id))
            self.mmsis.append(mmsi)

    
    def extract_kernel_params(self, model):
        params = {}
        kernels = model.covar_module.data_covar_module.kernels
        
        # RBF kernel assumed at index 0, Linear at index 1
        # params['rbf_lengthscale'] = kernels[0].lengthscale.item()
        # params['linear_variance'] = kernels[1].variance.item()


        # Extract the specific kernel parameters as before
        try:
            kernels = model.covar_module.data_covar_module.kernels
            # RBF kernel assumed at index 0, Linear at index 1
            params['rbf_lengthscale'] = kernels[0].lengthscale.item()
            params['linear_variance'] = kernels[1].variance.item()
        except Exception as e:
            print(f"Error extracting specific kernel parameters: {str(e)}")
            # Provide fallback values
            params['rbf_lengthscale'] = 1.0
            params['linear_variance'] = 1.0
        
        # Add all model parameters
        for param_name, param in model.named_parameters():
            # Handle different parameter shapes
            if param.numel() == 1:  # Single value parameter
                params[f"param_{param_name.replace('.', '_')}"] = param.item()
            else:  # Multi-dimensional parameter
                # For vectors, we can add each element separately
                if param.dim() == 1 and param.numel() <= 10:  # Reasonable size vector
                    for i, val in enumerate(param.tolist()):
                        params[f"param_{param_name.replace('.', '_')}_{i}"] = val
                else:
                    # For larger tensors, we can use statistics
                    params[f"param_{param_name.replace('.', '_')}_mean"] = param.mean().item()
                    params[f"param_{param_name.replace('.', '_')}_std"] = param.std().item()
                    params[f"param_{param_name.replace('.', '_')}_min"] = param.min().item()
                    params[f"param_{param_name.replace('.', '_')}_max"] = param.max().item()
        # print(len(params), "params extracted from model")
        
        return params
    
    def get_parameter_names(self):
        """
        Returns a list of parameter names that are extracted from each model.
        
        Returns:
            list: List of parameter names
        """
        if not hasattr(self, '_param_names') or self._param_names is None:
            # Generate parameter names by extracting from a model
            if self.models:
                first_model = next(iter(self.models.values()))
                params = self.extract_kernel_params(first_model)
                self._param_names = list(params.keys())
            else:
                self._param_names = []
                
        return self._param_names
    
    def get_unique_group_ids(self):
        """
        Returns a list of unique group_ids in the dataset.
        """
        return list(set(group_id for _, _, group_id in self.data))
    
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mmsi, kernel_params, group_id = self.data[idx]

        # Unscale kernel params if scalers are provided
        rbf_lengthscale = kernel_params[0].item()
        linear_variance = kernel_params[1].item()

        if self.scalers_by_mmsi is not None and mmsi in self.scalers_by_mmsi:
            scaler_dict = self.scalers_by_mmsi[mmsi]
            # Unscale rbf_lengthscale using time_scaler (for StandardScaler)
            if 'time_scaler' in scaler_dict:
                time_scaler = scaler_dict['time_scaler']
                # For StandardScaler, unscale: x_real = x_scaled * scale_ + mean_
                rbf_lengthscale_unscaled = rbf_lengthscale * time_scaler.scale_[0]
            else:
                rbf_lengthscale_unscaled = rbf_lengthscale

            # Unscale linear_variance using state_scaler (for StandardScaler)
            if 'state_scaler' in scaler_dict:
                state_scaler = scaler_dict['state_scaler']
                # For StandardScaler, variance scales as (scale_)^2
                linear_variance_unscaled = linear_variance * (state_scaler.scale_[0] ** 2)
            else:
                linear_variance_unscaled = linear_variance

            kernel_params_unscaled = torch.tensor(
                [rbf_lengthscale_unscaled, linear_variance_unscaled],
                dtype=kernel_params.dtype,
                device=kernel_params.device
            )
        else:
            kernel_params_unscaled = kernel_params

        return mmsi, kernel_params.to(self.device), group_id
        # return 0

