from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.sparse import issparse
import h5py


class SingleCellDataset(Dataset):
    """
    Dataset for dataloader
    """
    def __init__(self, adata):
        self.adata = adata
        self.shape = adata.shape
        
    def __len__(self):
        return self.adata.X.shape[0]
    
    def __getitem__(self, idx):
        if issparse(self.adata.X):
            x = self.adata.X[idx].toarray().squeeze()
        else:
            x = self.adata.X[idx].squeeze()
            
        return x


def create_dataset_loader(
        adata,
        batch_size=64, 
        log=None
    ):
    """
    Load dataset with preprocessing
    """

    if log: log.info('Raw dataset shape: {}'.format(adata.shape))
    
    if 'batch' not in adata.obs:
        adata.obs['batch'] = 'batch'
    if adata.obs['batch'] is not None:
        adata.obs['batch'] = adata.obs['batch'].astype('category')

    scdata = SingleCellDataset(adata) # Wrap AnnData into Pytorch Dataset
    train_loader = DataLoader(
        scdata, 
        batch_size=batch_size, 
        drop_last=True, 
        shuffle=True, 
        num_workers=16
    )
    
    predict_loader = DataLoader(scdata, batch_size=batch_size, drop_last=False, shuffle=False)
    
    return train_loader, predict_loader


def read_h5ad_to_dask(filename):
    adata = sc.read_h5ad(filename)
    
    return adata


class CellDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return sum([h5py.File(path, 'r')['data'].shape[0] for path in self.file_paths])

    def __getitem__(self, idx):
        file_idx = 0
        while idx >= h5py.File(self.file_paths[file_idx], 'r')['data'].shape[0]:
            idx -= h5py.File(self.file_paths[file_idx], 'r')['data'].shape[0]
            file_idx += 1
        with h5py.File(self.file_paths[file_idx], 'r') as f:
            data = f['data'][idx]
            if self.transform:
                data = self.transform(data)
            return data
