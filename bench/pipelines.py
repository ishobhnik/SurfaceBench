from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import numpy as np
import time
import warnings
from surface_dataclasses import Problem, SearchResult, Equation
from searchers.base import BaseSearcher
from utils import (
    get_surface_callable_from_llm_code,
    generate_points_from_surface_callable,
    compute_chamfer_distance,
    compute_hausdorff_distance,
)

def compute_surface_metrics(
    predicted_equation: Equation, 
    gt_equation: Equation,        
    problem_samples: np.ndarray,  
    surface_type: str,            
    num_points_for_pc: int = 5000
) -> Dict[str, Any]:
    """
    Computes Chamfer and Hausdorff distances between the ground truth and predicted surfaces.
    
    Args:
        predicted_equation: The Equation object discovered by the searcher.
        gt_equation: The ground truth Equation object.
        problem_samples: A (N, 3) NumPy array of ground truth sampled points for evaluation.
                         These will be compared against the generated predicted point cloud.
        surface_type: The type of surface ('explicit', 'parametric', 'implicit').
        num_points_for_pc: Number of points to generate for the predicted point cloud.
    """
    metrics = {
        "chamfer_distance": float('inf'),
        "hausdorff_distance": float('inf'),
        "num_generated_points": 0,
        "is_valid_expression": False,
        "error_message": None
    }
    try:
        predicted_callable = get_surface_callable_from_llm_code(
            predicted_equation.program_format,
            surface_type
        )
        if predicted_callable is None:
            metrics["error_message"] = "Failed to create callable from predicted expression."
            return metrics
        metrics["is_valid_expression"] = True

    except Exception as e:
        metrics["error_message"] = f"Error in callable creation: {e}"
        return metrics

    predicted_points_cloud = None
    if surface_type in ['explicit', 'parametric']:
        if problem_samples.shape[0] > 0:
            min_coords = np.min(problem_samples, axis=0)
            max_coords = np.max(problem_samples, axis=0)
            
            param_ranges = {}
            if surface_type == 'explicit':
                param_ranges['x'] = [min_coords[0], max_coords[0]]
                param_ranges['y'] = [min_coords[1], max_coords[1]]
            elif surface_type == 'parametric':
                 pass 
            
        predicted_points_cloud = generate_points_from_surface_callable(
            predicted_callable,
            surface_type,
            num_points=num_points_for_pc,
            param_ranges=param_ranges 
        )
    elif surface_type == 'implicit':
        
        if problem_samples.shape[0] > 0: 
            min_coords = np.min(problem_samples, axis=0)
            max_coords = np.max(problem_samples, axis=0)
            param_ranges = {
                'x': [min_coords[0], max_coords[0]],
                'y': [min_coords[1], max_coords[1]],
                'z': [min_coords[2], max_coords[2]]
            }
            predicted_points_cloud = generate_points_from_surface_callable(
                predicted_callable,
                surface_type,
                num_points=num_points_for_pc,
                param_ranges=param_ranges
            )
        else:
             warnings.warn("No ground truth samples available for implicit surface to infer ranges.")
             predicted_points_cloud = None


    if predicted_points_cloud is None or predicted_points_cloud.shape[0] == 0:
        metrics["error_message"] = "Failed to generate valid point cloud from predicted equation."
        return metrics
    
    metrics["num_generated_points"] = predicted_points_cloud.shape[0]

    metrics["chamfer_distance"] = compute_chamfer_distance(predicted_points_cloud, problem_samples)
    metrics["hausdorff_distance"] = compute_hausdorff_distance(predicted_points_cloud, problem_samples)

    return metrics


class EvaluationPipeline:
    def __init__(self):
        pass

    def run_and_evaluate(self, searcher: BaseSearcher, problem: Problem):
        start_time = time.time()
        search_results: List[SearchResult] = searcher.discover(problem.create_task())
        search_time = time.time() - start_time

        surface_type = "unknown"
        for prop in problem.gt_equation.symbol_properties:
            if prop.startswith("surface_type:"):
                surface_type = prop.split(":")[1]
                break
        
        if surface_type == "unknown":
            warnings.warn(f"Surface type not found in symbol_properties for problem {problem.equation_idx}.")

        gt_id_test_samples = problem.test_samples
        gt_ood_test_samples = problem.ood_test_samples

        outs = []
        for result in search_results:
            predicted_equation = result.equation
            
            id_surface_metrics = compute_surface_metrics(
                predicted_equation,
                problem.gt_equation, 
                gt_id_test_samples,
                surface_type
            )

            ood_surface_metrics = None
            if gt_ood_test_samples is not None and gt_ood_test_samples.size > 0:
                ood_surface_metrics = compute_surface_metrics(
                    predicted_equation,
                    problem.gt_equation,
                    gt_ood_test_samples,
                    surface_type
                )
            
            outs.append({
                "search_result": result,
                "search_time": search_time,
                "id_metrics": id_surface_metrics,
                "ood_metrics": ood_surface_metrics,
                **result.aux
            })

        return outs

    def evaluate_problems(self, 
                          problems: List[Problem], 
                          searcher: BaseSearcher, 
                          output_dir: Path,
                          result_file_subfix: str = ""):
        
        output_dir.mkdir(parents=True, exist_ok=True) 
        output_file_path = output_dir / f"results{result_file_subfix}.jsonl"
        
        visited_eqids = []
        if output_file_path.exists():
            visited_eqids = self.load_visited_problems(output_dir)

        for problem in problems:
            if problem.equation_idx in visited_eqids:
                print(f"Skipping problem: {problem.equation_idx} (gt: {problem.gt_equation.expression}) - already processed.")
                continue

            print(f"Finding equation for problem: {problem.equation_idx} (gt: {problem.gt_equation.expression})")
            
            outs = self.run_and_evaluate(searcher, problem)

            log_data = {
                'equation_id': problem.equation_idx,
                'gt_equation': problem.gt_equation.expression,
                'num_train_datapoints_llm_input': len(problem.train_samples), 
                'num_id_eval_datapoints': len(problem.test_samples),
                'num_ood_eval_datapoints': len(problem.ood_test_samples) if problem.ood_test_samples is not None else 0,
            }
            eval_results = []
            for out in outs:
                eval_results.append({
                    'search_time': out['search_time'],
                    'discovered_equation_str': out['search_result'].equation.expression,
                    'id_metrics': out['id_metrics'],
                    'ood_metrics': out['ood_metrics'],
                    **out['search_result'].aux
                })
            log_data['eval_results'] = eval_results
            
            with open(output_file_path, mode='a') as f:
                f.write(json.dumps(log_data, allow_nan=True) + "\n")

            visited_eqids.append(problem.equation_idx)
    
    @property
    def name(self):
        raise NotImplementedError

    def load_visited_problems(self, output_dir: Path) -> List[str]:
        """Loads IDs of problems already processed from jsonl files in output_dir."""
        result_files = list(output_dir.glob("results*.jsonl"))
        visited = []
        if len(result_files) > 0:
            for result_file in result_files:
                try:
                    with open(result_file, 'r') as f:
                        for line in f.readlines():
                            visited.append(json.loads(line)['equation_id']) 
                except json.JSONDecodeError as e:
                    warnings.warn(f"Skipping malformed result file {result_file}: {e}")
            visited = list(visited)
        return visited