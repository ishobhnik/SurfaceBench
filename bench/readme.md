## Dataset Preparation Script (`prepare_surface_data.py`)

This script helps convert your raw `.npz` sampled data into the HDF5 format with train/test/OOD splits and generates the corresponding JSON metadata.

1.  **Run the script**: Navigate to your project root and run:
    ```bash
    python prepare_surface_data.py
    ```
2. **Expected Output**: The script will create a `surface_data.h5` file in the `bench/data/` directory and a `metadata.json` file in the same directory. These files contain the processed data and metadata for your 3D surface equations.

# Bench: The SURFACEBENCH Evaluation Framework

This directory contains the core components for the **SURFACEBENCH** evaluation framework. It is an adaptation of the original LLM-SRBench to handle the unique challenges of 3D surface equation discovery.

The purpose of these files is to provide a standardized, reusable pipeline for loading our specific 3D surface dataset and evaluating the performance of LLM-based symbolic regression methods.

## Directory Contents

* `surface_dataclasses.py`: Defines the Python data structures for our `Problem`, `Equation`, and `SearchResult` objects.
* `datamodules.py`: The data loading module. It connects to our Hugging Face dataset, downloads the metadata and HDF5 data files, and loads them into memory as `Problem` objects.
* `pipelines.py`: The top-level evaluation orchestrator. It runs the full evaluation loop, passing problems to a searcher, and logging the results using a standard format.
* `utils.py`: A collection of utility functions crucial for 3D surface evaluation, including point cloud generation from a given equation and computing geometric metrics like Chamfer and Hausdorff distance.
* `searchers/base.py`: The abstract base class for all searcher implementations.

## How to Verify Core Functionality

Before running a full end-to-end experiment, you can verify that the data loading and evaluation metrics are working correctly with these simple tests.

### Prerequisites

Make sure you have all the necessary Python packages installed:
```bash
pip install numpy scipy scikit-learn h5py huggingface-hub datasets openai transformersx
```
1. Test the Data Loading (datamodules.py)
This test ensures that your datamodules.py is correctly configured to download and load a problem from your Hugging Face repository.

2. Navigate to the `bench` directory:
```bash
cd path/to/your/project/bench
```
3. Run the script directly:
```bash
python datamodules.py
```

Expected Output: The script should print a log of the dataset being downloaded and a confirmation that at least one problem was loaded, along with the shapes of its train_samples, ID Test samples, and OOD Test samples

4. Test the Evaluation Utilities (utils.py)
This test verifies that the utils.py functions for generating point clouds and computing geometric metrics are working as expected.

 Navigate to the bench/ directory:
```bash
cd path/to/your/project/bench
```
 Run the script directly:
```bash
python utils.py
```
Expected Output: The script should show tests running for explicit, parametric, and implicit surfaces. You should see Generated...PC shape lines with correct dimensions ((2500, 3)) and small positive values for Chamfer Distance and Hausdorff Distance for the example cases.

## How to Run a Full Experiment
A full experiment requires the llmsr folder to be properly configured. Once the llmsr parts are adapted, you can run an end-to-end evaluation using eval.py script in the project root. This script orchestrates the bench/pipelines.py to run the LLMSR searcher for a given dataset and logs the results.