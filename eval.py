import os
import sys
import yaml
from pathlib import Path
import multiprocessing
from datetime import datetime
from argparse import ArgumentParser, Namespace
from dotenv import load_dotenv

project_root_dir = os.path.dirname(os.path.abspath(__file__))
methods_dir = os.path.join(project_root_dir, "methods/llmsr")
if methods_dir not in sys.path:
    sys.path.insert(0, methods_dir)

# Import components from bench folder
from bench.datamodules import get_datamodule
from bench.pipelines import EvaluationPipeline

# These imports are now relative to the 'methods' directory we added to sys.path
from methods.llmsr.searcher import LLMSRSearcher
from methods.llmsr import config as llmsr_config_module
from methods.llmsr import sampler as llmsr_sampler_module


def main():
    load_dotenv()

    parser = ArgumentParser(description="Run SURFACEBENCH evaluation pipeline.")
    parser.add_argument("--searcher_config", 
                        required=True, 
                        help="Path to the YAML configuration file for the searcher (e.g., configs/llmsr_searcher.yaml)")
    parser.add_argument("--dataset", 
                        required=True, 
                        help="Name of the dataset to use (e.g., 'surfacebench')")
    parser.add_argument("--ds_root_folder", 
                        default=None, 
                        help="Root folder for local datasets (not typically used for HF datasets).")
    parser.add_argument("--resume_from", 
                        default=None, 
                        help="Path to a directory to resume an experiment from.")
    parser.add_argument("--problem_name", 
                        default=None, 
                        help="Specific problem name to evaluate (e.g., 'Hybrid_Dual_Domain_01'). Evaluates all if None.")
    parser.add_argument("--local_llm_port", 
                        default=None, type=int, 
                        help="Port for local LLM server (if api_type is 'vllm').")
    
    args = parser.parse_args()

    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S_%f")

    print(f"Initializing data module for dataset: {args.dataset}")
    dm = get_datamodule(name=args.dataset, root_folder=args.ds_root_folder)
    dm.setup()

    print(f"Loading searcher configuration from: {args.searcher_config}")
    with open(args.searcher_config, 'r') as f:
        searcher_cfg_dict = yaml.safe_load(f)
    searcher_cfg = Namespace(**searcher_cfg_dict)

    if args.resume_from is None:
        output_path = Path(f"logs/{dm.name}/{searcher_cfg.name}/{now_str}")
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"New experiment logs will be saved to: {output_path}")
    else:
        output_path = Path(args.resume_from)
        if not output_path.exists():
            raise FileNotFoundError(f"Resume path not found: {output_path}")
        print(f"Resuming experiment from: {output_path}")
    
    searcher_log_path = output_path / "search_logs"
    searcher_log_path.mkdir(exist_ok=True, parents=True)

    print("\nConfiguring LLM and Searcher...")

    api_key_from_env = None
    if searcher_cfg.api_type == "hfinf":
        api_key_from_env = os.getenv('HFINF_API_KEY')
    elif searcher_cfg.api_type == "vllm":
        api_key_from_env = os.getenv('VLLM_API_KEY')
        if args.local_llm_port: 
            searcher_cfg.api_url = searcher_cfg.api_url.format(args.local_llm_port)
    elif searcher_cfg.api_type == "openai":
        api_key_from_env = os.getenv('OPENAI_API_KEY')

    llm_config_instance = llmsr_config_module.Config(
        experience_buffer=llmsr_config_module.ExperienceBufferConfig(
            functions_per_prompt=searcher_cfg.functions_per_prompt,
            num_islands=searcher_cfg.num_islands
        ),
        num_samplers=searcher_cfg.num_samplers,
        num_evaluators=searcher_cfg.num_evaluators,
        samples_per_prompt=searcher_cfg.samples_per_prompt,
        evaluate_timeout_seconds=searcher_cfg.evaluate_timeout_seconds,
        use_api= searcher_cfg.api_type not in ['local', 'vllm'],
        api_model=searcher_cfg.api_model,
        api_key=api_key_from_env,
        api_url=searcher_cfg.api_url,
    )

    SamplerClass_to_use = llmsr_sampler_module.LocalLLM

    # Initialize the LLMSRSearcher
    searcher = LLMSRSearcher(
        name=searcher_cfg.name,
        cfg=llm_config_instance, 
        SamplerClass=SamplerClass_to_use,
        global_max_sample_num=searcher_cfg.global_max_sample_num,
        log_path=str(searcher_log_path)
    )

    problems_to_evaluate = dm.problems
    if args.problem_name is not None:
        problems_to_evaluate = list(filter(lambda p: p.equation_idx == args.problem_name, problems_to_evaluate))
        if not problems_to_evaluate:
            print(f"Warning: Problem '{args.problem_name}' not found in the loaded dataset.")
            sys.exit(0)
    
    print(f"Total number of problems selected for evaluation: {len(problems_to_evaluate)}")

    print("\nInitializing Evaluation Pipeline...")
    eval_pipeline = EvaluationPipeline()

    print("\nStarting LLMSR Searcher and Evaluation for problems...")
    eval_pipeline.evaluate_problems(
        problems=problems_to_evaluate,
        searcher=searcher,
        output_dir=output_path, 
        result_file_subfix="_surfacebench"
    )

    print("\n--- SURFACEBENCH LLMSR Pipeline Run Complete ---")
    print(f"Detailed logs in: {output_path}/search_logs/")
    print(f"Overall results in: {output_path}/results_surfacebench.jsonl")


if __name__ == "__main__":
    # This is for Windows compatibility with multiprocessing
    multiprocessing.freeze_support() 
    main()