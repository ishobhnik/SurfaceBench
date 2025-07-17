import os
import sys
import yaml
from pathlib import Path
import multiprocessing
from datetime import datetime
from bench.datamodules import get_datamodule
from bench.pipelines import EvaluationPipeline
from argparse import ArgumentParser, Namespace
from dotenv import load_dotenv

def main():
    load_dotenv()

    parser = ArgumentParser()
    parser.add_argument("--searcher_config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ds_root_folder", default=None)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--problem_name", default=None)
    parser.add_argument("--local_llm_port", default=None, type=int)
    args = parser.parse_args()

    now = datetime.now()
    now_str = now.strftime("%m-%d-%Y_%H-%M-%S-%f")

    dm = get_datamodule(name=args.dataset, root_folder=args.ds_root_folder)
    dm.setup()

    with open(args.searcher_config) as f:
        searcher_cfg = yaml.safe_load(f)
    searcher_cfg = Namespace(**searcher_cfg)

    if args.resume_from is None:
        output_path = Path(f"logs/{dm.name}/{searcher_cfg.name}/{now_str}")
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(args.resume_from)
    searcher_log_path = output_path / "search_logs"
    searcher_log_path.mkdir(exist_ok=True, parents=True)

    temp_dir = Path("logs/tmp")
    temp_dir.mkdir(exist_ok=True, parents=True)

    if searcher_cfg.api_type == "hfinf":
        api_key = os.environ['HFINF_API_KEY']
    elif searcher_cfg.api_type == "vllm":
        api_key = os.environ['VLLM_API_KEY']
        searcher_cfg.api_url = searcher_cfg.api_url.format(args.local_llm_port)
    elif searcher_cfg.api_type == "openai":
        api_key = os.environ['OPENAI_API_KEY']
    else:
        api_key = None

    if searcher_cfg.class_name == 'LLMSRSearcher':
        sys.path.append(os.path.join(os.path.dirname(__file__), "methods"))
        from methods.llmsr.searcher import LLMSRSearcher
        from methods.llmsr import config, sampler

        exp_conf = config.ExperienceBufferConfig(
            num_islands=searcher_cfg.num_islands
        )
        cfg = config.Config(
            experience_buffer=exp_conf,
            use_api = searcher_cfg.api_type != 'local',
            api_model = searcher_cfg.api_model,
            samples_per_prompt = searcher_cfg.samples_per_prompt,
        )
        sampler_class = lambda samples_per_prompt: sampler.LocalLLM(
            samples_per_prompt=samples_per_prompt,
            local_llm_url=searcher_cfg.api_url,
            api_url=searcher_cfg.api_url,
            api_key=api_key,
        )
        searcher = LLMSRSearcher(
            searcher_cfg.name,
            cfg,
            sampler_class,
            global_max_sample_num=searcher_cfg.global_max_sample_num,
            log_path=searcher_log_path
        )
    else:
        raise ValueError("Unknown searcher class")

    problems = dm.problems
    if args.problem_name is not None:
        problems = list(filter(lambda p: p.equation_idx == args.problem_name, problems))
    print(f"Total number of problems: {len(problems)}")

    pipeline = EvaluationPipeline()
    pipeline.evaluate_problems(
        problems,
        searcher,
        output_path
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()