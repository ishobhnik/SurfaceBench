# LLM-SRBench

This is the official repository for the paper "LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models"


```
project
│   README.md
|   eval.py
|   .env
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

3. Set correct values for environment variables in `.env` file. Copy `.env.example` to `.env` and set:
   - `VLLM_API_KEY`: API key for local vLLM server (e.g. 'token-abc123')
   - `OPENAI_API_KEY`: OpenAI API key if using OpenAI models
   - `SGA_PYTHON_PATH`: Path to Python executable in your SGA conda environment

4. Run the `eval.py` script with the following arguments:

   - `--searcher_config`: Path to YAML config file for the searcher (required)
   - `--dataset`: Name of dataset to evaluate (required)
   - `--ds_root_folder`: Root folder containing dataset files (optional)
   - `--resume_from`: Path to previous run directory to resume from (optional)
   - `--problem_name`: Name of specific problem to evaluate (optional)
   - `--local_llm_port`: Port number for local LLM server (optional)
    
Options for datasets are:
* feynman
* lsrtransform (lsr-transform)
* matsci (lsr-synth)
* chem_react (lsr-synt)
* phys_osc (lsr-synth)
* bio_pop_growth (lsr-synth)

The run will create log files at the `logs` folder. You can resume your run with option `--resume_from <log_dir>`. For example, 
`--resume_from logs/MatSci/llmsr_4_10_10/01-16-2025_17-41-04-540953`. This will skip finished equations.

Examples of usage with a local LLM:
```python
# Starting VLLM server
CUDA_VISIBLE_DEVICES=5 vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype auto --api-key token-abc123 --port 10005

# LSR-Transform Dataset
python eval.py --dataset lsrtransform --searcher_config configs/llmdirect_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset lsrtransform --searcher_config configs/lasr_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset lsrtransform --searcher_config configs/sga_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset lsrtransform --searcher_config configs/llmsr_llama31_8b.yaml --local_llm_port 10005

# Bio-Pop-Growth Dataset
python eval.py --dataset bio_pop_growth --searcher_config configs/llmdirect_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset bio_pop_growth --searcher_config configs/lasr_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset bio_pop_growth --searcher_config configs/sga_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset bio_pop_growth --searcher_config configs/llmsr_llama31_8b.yaml --local_llm_port 10005

# Chem React Kinetics
python eval.py --dataset chem_react --searcher_config configs/llmdirect_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset chem_react --searcher_config configs/lasr_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset chem_react --searcher_config configs/sga_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset chem_react --searcher_config configs/llmsr_llama31_8b.yaml --local_llm_port 10005

# Matsci SS
python eval.py --dataset matsci --searcher_config configs/llmdirect_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset matsci --searcher_config configs/lasr_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset matsci --searcher_config configs/sga_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset matsci --searcher_config configs/llmsr_llama31_8b.yaml --local_llm_port 10005

# Phys oscillator
python eval.py --dataset phys_osc --searcher_config configs/llmdirect_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset phys_osc --searcher_config configs/lasr_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset phys_osc --searcher_config configs/sga_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset phys_osc --searcher_config configs/llmsr_llama31_8b.yaml --local_llm_port 10005
```