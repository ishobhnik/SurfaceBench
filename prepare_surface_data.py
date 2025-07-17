import os
import numpy as np
import h5py
from sklearn.model_selection import train_test_split

def load_points(npz_path: str) -> np.ndarray:
    """Load x, y, z arrays from an .npz file and stack into (N, 3) points."""
    data = np.load(npz_path)
    x, y, z = data['x'], data['y'], data['z']
    return np.stack((x, y, z), axis=-1)

def split_points(
    points: np.ndarray,
    prompt_size: int = 20,
    ood_ratio: float = 0.20,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split points into:
      - llm_prompt: first prompt_size samples
      - id_test: (1 - ood_ratio) of the remainder
      - ood_test: ood_ratio of the remainder
    """
    rng = np.random.default_rng(seed)
    # First split off prompt points
    if points.shape[0] > prompt_size:
        llm_prompt, remainder = train_test_split(
            points,
            train_size=prompt_size,
            random_state=seed
        )
    else:
        llm_prompt, remainder = points, np.empty((0, 3))
    # Split remainder into ID and OOD
    if remainder.shape[0] > 0:
        id_test, ood_test = train_test_split(
            remainder,
            test_size=ood_ratio,
            random_state=seed
        )
    else:
        id_test, ood_test = np.empty((0, 3)), np.empty((0, 3))
    return llm_prompt, id_test, ood_test

def save_to_hdf5(
    hdf5_path: str,
    group_name: str,
    train_data: np.ndarray,
    id_test_data: np.ndarray,
    ood_test_data: np.ndarray
) -> None:
    """Save the three datasets under the specified group in an HDF5 file."""
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    with h5py.File(hdf5_path, 'a') as f:
        if group_name in f:
            del f[group_name]
        grp = f.create_group(group_name)
        grp.create_dataset('train_data', data=train_data)
        grp.create_dataset('id_test_data', data=id_test_data)
        grp.create_dataset('ood_test_data', data=ood_test_data)

def main():
    # Configuration
    npz_path = 'Hybrid Multi-Modal Symbolic Surfaces/Sampled_Data/Hybrid_Surface_Contact_Hybrid_samples.npz'
    hdf5_path = 'data/SurfaceBenchData.hdf5'
    hdf5_group = '/explicit/Hybrid_Dual_Domain_01'
    prompt_size = 20
    ood_ratio = 0.20

    # Load and split
    points = load_points(npz_path)
    llm_prompt, id_test, ood_test = split_points(points, prompt_size, ood_ratio)

    # Save to HDF5
    save_to_hdf5(hdf5_path, hdf5_group, llm_prompt, id_test, ood_test)

    # Summary
    print(f"Loaded {points.shape[0]} points")
    print(f"  • Prompt:   {llm_prompt.shape}")
    print(f"  • ID test:  {id_test.shape}")
    print(f"  • OOD test: {ood_test.shape}")
    print(f"Data written to {hdf5_path} under group {hdf5_group}")

if __name__ == '__main__':
    main()
