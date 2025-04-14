# `LLM-SRBench`: A New Benchmark for Scientific Equation Disscovery (Symbolic Regression) with LLMs

This is the official repository for the paper "LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models"

[![Paper](https://img.shields.io/badge/arXiv-XXX.XXX-b31b1b.svg)](XXX.XXX)
[![Benchmark Dataset](https://img.shields.io/badge/🤗_Hugging_Face-nnheui/llm--srbench-ffbd00.svg)](https://huggingface.co/datasets/nnheui/llm-srbench)




![](images/task_sed.png)

## Overview
In this paper, we introduce LLM-SRBench, a comprehensive benchmark with $239$ challenging problems across four scientific domains specifically designed to evaluate LLM-based scientific equation discovery methods while preventing trivial memorization.
Our benchmark comprises two main categories: LSR-Transform, which transforms common physical models into less common mathematical representations to test reasoning beyond memorization,
and LSR-Synth, which introduces synthetic, discovery-driven problems requiring data-driven reasoning.

## Updates

* **13 Apr, 2025**: Primary release

## Get Started

### Installation

To run the code, create a conda environment and install the dependencies provided in the `requirements.txt` or `environment.yml`:

```
conda create -n llmsrbench python=3.11.7
conda activate llmsrbench
pip install -r requirements.txt
```

Note: Requires Python ≥ 3.9

You also need to install other packages for each search method from their original github repositories.
  - [llmsr](https://github.com/deep-symbolic-mathematics/LLM-SR)
  - [lasr](https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl)
  - [SGA](https://github.com/PingchuanMa/SGA)


### Datasets

The data for the benchmark will be automatically downloaded.

### Supported methods

We provide implementation for [llmsr](https://github.com/deep-symbolic-mathematics/LLM-SR), [lasr](https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl), [SGA](https://github.com/PingchuanMa/SGA) in the `methods` folder.

In order to include a new method, please refer to the implementing section for detailed instructions on how to add a new search method to the project. This includes setting up the necessary configurations, implementing the searcher class, and ensuring compatibility with the existing framework.

### How to runs
1. Activate the appropriate conda environment.
2. Launch a local LLM server. While our implementation utilizes vllm, you can opt for other libraries as long as you implement the necessary functionality in the searcher class. For example, to start the server with the vllm library, use the command: `vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype auto --api-key token-abc123 --port 10005`.

3. Configure the environment variables in the `.env` file. Duplicate `.env.example` to `.env` and specify the following:
   - `VLLM_API_KEY`: Your API key for the local vLLM server (e.g., 'token-abc123').
   - `OPENAI_API_KEY`: Your OpenAI API key if you are utilizing OpenAI models.
   - `SGA_PYTHON_PATH`: The path to the Python executable in your SGA conda environment if you are using the SGA searcher.

4. Execute the `eval.py` script with the required arguments:
   - `--searcher_config`: Path to the YAML configuration file for the searcher (mandatory).
   - `--dataset`: The name of the dataset to evaluate (mandatory).
   - `--resume_from`: The path to a previous run directory to continue from (optional).
   - `--problem_name`: The specific problem name to evaluate (optional).
   - `--local_llm_port`: The port number for the local LLM server (optional).

   Available dataset options include:
   * lsrtransform (lsr-transform)
   * matsci (lsr-synth)
   * chem_react (lsr-synth)
   * phys_osc (lsr-synth)
   * bio_pop_growth (lsr-synth)

The execution will generate log files in the `logs` folder. You can resume your run using the `--resume_from <log_dir>` option. For instance, 
`--resume_from logs/MatSci/llmsr_4_10_10/01-16-2025_17-41-04-540953` will bypass already completed equations.

The working directory structure will be as follows:

```
project
│   README.md
|   eval.py
|   .env
└───bench/
|
└───methods/
|   └───direct
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

### Evaluation scripts

Please take a look at `example_script.sh` for examples of usage with a local LLM.

## Implementing a new searcher

To implement a new searcher, you must create a class that inherits from the base class `BaseSearcher`. This base class provides the foundational structure for your searcher, including essential methods that need to be overridden.

```python
class BaseSearcher:
    def __init__(self, name) -> None:
        self._name = name

    def discover(self, task: SEDTask) -> List[SearchResult]:
        '''
        Return:
            List of SearchResult
        '''
        raise NotImplementedError

    def __str__(self):
        return self._name
```

The input `task` will provide a description of the target equation, description of input variables, and training data points.

An example of searcher is
```python
class DirectPromptingSearcher(BaseSearcher):
    def __init__(self, name, num_sample, api_type, api_model, api_url):
        super().__init__(name)
        self.num_samples = num_samples
        self.llm = LLM(api_type, api_model, api_url)

    def discover(self, task: SEDTask):
        dataset = task.samples
        symbol_descs = task.symbol_descs

        prompt = f"Find the mathematical function skeleton that represents {symbol_descs[0]}, given data on {", ".join(symbol_descs[1:-1]) + ", and " + symbol_descs[-1]}"
        
        best_program, best_score = None, -np.inf
        for _ in range(self.num_samples):
            program_str, aux = self.llm.sample_program(prompt)
            score = evaluate(program_str, dataset)
            if score > best_score:
                best_program = program_str

        best_equation = Equation(
            symbols=info["symbols"],
            symbol_descs=info["symbol_descs"],
            symbol_properties=info["symbol_properties"],
            expression=None,
            program_format = best_program,
            lambda_format = programstr2lambda(best_program)
        )

        return [
            SearchResult(
                equation=best_equation,
                aux=aux
            )
        ]
```

Once you’ve implemented your searcher, create a corresponding configuration file in the configs folder. For example:

```yaml
name: DirectPrompting-Llama31_8b
class_name: DirectPromptingSearcher
api_type: "vllm"
api_model: "meta-llama/Llama-3.1-8B-Instruct"
api_url: "http://localhost:{}/v1/"
num_samples: 1000
```

To evaluate with this searcher, run eval.py and provide the path to its configuration file; this will load the settings and initiate the evaluation process.

## Citation

## License

This repository is licensed under MIT licence.

This work is built on top of other open source projects, including [LLM-SR](https://github.com/deep-symbolic-mathematics/llm-sr), [LaSR](https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl), [SGA](https://github.com/PingchuanMa/SGA) and [PySR](https://github.com/MilesCranmer/PySR). We thank the original contributors of these works for open-sourcing their valuable source codes. 


## Contact Us

For any questions or issues, you are welcome to open an issue in this repo, or contact us at parshinshojaee@vt.edu, and ngochieutb13@gmail.com .
