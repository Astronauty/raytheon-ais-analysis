import torch
from torch.utils.data import Dataset


class GPKernelShipClassificationDataset(Dataset):
    """
    Dataset for ship classification using Gaussian Process kernel parameters.
    """
    def __init__(self, gp_regression_dataset, models, device):
        """_summary_

        Args:
            gp_regression_dataset (_type_): Dataset containing GP regression data. (Used to extract MMSI and class labels)
            models (_type_): Dictionary of fitted GP models per MMSI, where keys are MMSI and values are GP models.
            device (_type_): _description_
        """
        self.device = device
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
        return mmsi, kernel_params.to(self.device), group_id
        # return 0