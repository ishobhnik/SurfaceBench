import json
import numpy as np
import h5py
import datasets
import warnings
from typing import Optional, Any, List
from surface_dataclasses import Equation, Problem, SEDTask
from pathlib import Path
from huggingface_hub import snapshot_download

HF_REPO_ID = "Shobhnik/SurfaceBench"
METADATA_FILE_IN_HF = "Hybrid Multi-Modal Symbolic Surfaces-Eq1.json"
HDF5_DATA_FILE_IN_HF = "SurfaceBenchData.hdf5"
HDF5_BASE_GROUP = "/explicit"

class SurfaceBenchDataModule:
    """
        Data module to load the SURFACEBENCH dataset from a Hugging Face repository.
    """
    def __init__(self):
        self._dataset_dir = None
        self._name = "SurfaceBench"
        self.problems: List[Problem] = []
        self.name2id = {}

    def setup(self):
        """
        Downloads the dataset snapshot from Hugging Face and loads problems
        based on the metadata JSON and HDF5 data.
        """
        try:
            # Downloads the entire dataset snapshot from Hugging Face to a local cache
            self._dataset_dir = Path(snapshot_download(repo_id=HF_REPO_ID, repo_type="dataset"))
            print(f"Dataset downloaded to: {self._dataset_dir}")
        except Exception as e:
            print("Please ensure you have logged in `huggingface-cli login` and the repo ID is correct.")
            return 

        metadata_file_path = self._dataset_dir / METADATA_FILE_IN_HF
        if not metadata_file_path.exists():
            raise FileNotFoundError(f"Metadata file not found in HF snapshot: {metadata_file_path}. Expected at: {metadata_file_path}")
        
        with open(metadata_file_path, 'r') as f:
            metadata_list = json.load(f)

        sample_h5file_path = self._dataset_dir / HDF5_DATA_FILE_IN_HF
        if not sample_h5file_path.exists():
            raise FileNotFoundError(f"HDF5 data file not found in HF snapshot: {sample_h5file_path}. Expected at: {sample_h5file_path}")
        
        with h5py.File(sample_h5file_path, "r") as sample_file:
            for e_data in metadata_list:
                hdf5_data_path = f"{HDF5_BASE_GROUP}/{e_data['name']}"
                
                if hdf5_data_path not in sample_file:
                    warnings.warn(f"HDF5 group '{hdf5_data_path}' not found in '{HDF5_DATA_FILE_IN_HF}'. Skipping problem {e_data['name']}.")
                    continue

                # Load the 'train_data', 'id_test_data', 'ood_test_data' arrays from HDF5
                samples = {k:v[...].astype(np.float64) for k,v in sample_file[hdf5_data_path].items()}

                gt_symbols = e_data['symbols']
                gt_equation = Equation(
                    symbols=gt_symbols,
                    symbol_descs=e_data.get('symbol_descs', []),
                    symbol_properties=e_data.get('symbol_properties', []),
                    expression=e_data['expression'], 
                    desc=e_data.get('desc', '')
                )

                self.problems.append(Problem(
                    dataset_identifier=self._name,
                    equation_idx=e_data['name'],
                    gt_equation=gt_equation,
                    samples=samples
                ))
        
        self.name2id = {p.equation_idx: i for i,p in enumerate(self.problems)}
        print(f"Successfully loaded {len(self.problems)} SURFACEBENCH problems from Hugging Face data.")

    @property
    def name(self):
        return self._name
def get_datamodule(name, root_folder=None):
    if name == 'surfacebench':
        return SurfaceBenchDataModule()
    else:
        raise ValueError(f"Unknown datamodule name: {name}. Only 'surfacebench' is supported.")
    

# if __name__ == "__main__":
#     datamodule = get_datamodule("surfacebench")
#     datamodule.setup()

#     print(f"Total problems loaded: {len(datamodule.problems)}")