import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class GPKernelShipClassificationDataset(Dataset):
    """
    Dataset for ship classification using Gaussian Process kernel parameters.
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
        """_summary_

        Args:
            model (_type_): _description_

        Returns:
            params (dictionary): _description_
        """
        params = {}
        kernels = model.covar_module.data_covar_module.kernels
        
        # RBF kernel assumed at index 0, Linear at index 1
        params['rbf_lengthscale'] = kernels[0].lengthscale.item()
        params['linear_variance'] = kernels[1].variance.item()
        
        return params
    
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