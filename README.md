# LLM-SRBench

This is the official repository for the paper "LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models"


```
project
│   README.md
|   eval.py
└───bench/
|
└───methods/
|   └───llmsr
|   └───lasr
|   └───sga_sr
|
└───datasets/
|
└───logs/
    └───<dataset-name>
        └───<method-name>
            └───<date>

```


1. Datasets are provided anonymously in [this drive link](https://drive.google.com/drive/folders/1TVhvzfR8eVD0bDpVmNcaHa_9hkgX0av3). Download them and put them at `datasets` directory.


2. Select the correct conda environment
    > pip install vllm python-dotenv zss datasets
    
    Install other packages for each search method from their original github repositories.

    - [llmsr](https://github.com/deep-symbolic-mathematics/LLM-SR)

    - [lasr](https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl)

    - [SGA](https://github.com/PingchuanMa/SGA)

    
Options for datasets are:
* feynman
* inv_feynman (lsr-transform)
* matsci (lsr-synth)
* chem_react (lsr-synt)
* phys_osc (lsr-synth)
* bio_pop_growth (lsr-synth)

The run will create log files at the `logs` folder. You can resume your run with option `--resume_from <log_dir>`. For example, 
`--resume_from logs/MatSci/llmsr_4_10_10/01-16-2025_17-41-04-540953`. This will skip finished equations.