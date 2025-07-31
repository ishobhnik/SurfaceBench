import os
import sys
import numpy as np
import warnings
from typing import List, Any, Tuple, Dict, Optional, Type
import re
import textwrap 

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from bench.surface_dataclasses import Problem, Equation, SearchResult, SEDTask
from bench.searchers.base import BaseSearcher 

from bench.utils import get_surface_callable_from_llm_code 

from methods.llmsr import pipeline, config as config_lib, sampler, evaluator 
from methods.llmsr import code_manipulation 

_DEFAULT_EVAL_RANGES = {
    'explicit': {'x': [-5.0, 5.0], 'y': [-5.0, 5.0]},
    'parametric': {'u': [0.0, 2*np.pi], 'v': [0.0, np.pi]},
    'implicit': {'x': [-5.0, 5.0], 'y': [-5.0, 5.0], 'z': [-5.0, 5.0]},
}
_NUM_POINTS_FOR_SANDBOX_PC = 2500 

# This dictionary maps surface types to their specific function signatures, etc.
_SURFACE_TYPE_INFO = {
    'explicit': {
        'input_args_str': 'x: np.ndarray, y: np.ndarray',
        'output_return_type': 'np.ndarray',
        'return_docstring': 'A numpy array representing the height (z) of the surface.',
        'function_name': 'get_z',
        'input_symbols_for_llm_code': ['x', 'y']
    },
    'parametric': {
        'input_args_str': 'u: np.ndarray, v: np.ndarray',
        'output_return_type': 'Tuple[np.ndarray, np.ndarray, np.ndarray]', 
        'return_docstring': 'A tuple of numpy arrays (x, y, z) representing the 3D coordinates.',
        'function_name': ['get_x', 'get_y', 'get_z'],
        'input_symbols_for_llm_code': ['u', 'v']
    },
    'implicit': {
        'input_args_str': 'x: np.ndarray, y: np.ndarray, z: np.ndarray',
        'output_return_type': 'np.ndarray',
        'return_docstring': 'A numpy array representing the implicit function value f(x,y,z).',
        'function_name': 'get_f',
        'input_symbols_for_llm_code': ['x', 'y', 'z']
    }
}


# These functions are copied here because the LLMSR sandbox environment is restricted
_INLINED_UTILS_FOR_SANDBOX = """
import numpy as np
import math
from scipy.spatial.distance import cdist 
from scipy.spatial import KDTree 

def _create_callable_from_code_string_sandbox(code_str, func_name, input_vars):
    local_env = {"math": math, "np": np}
    try:
        exec(code_str, local_env)
        func = local_env.get(func_name)
        if func and callable(func):
            return func
        return None
    except Exception as e:
        # warnings.warn(f"Sandbox: Error creating callable {func_name}: {e}") 
        return None

def _get_surface_callable_sandbox(llm_code_str, surface_type):
    if surface_type == 'explicit':
        get_z_func = _create_callable_from_code_string_sandbox(llm_code_str, 'get_z', ['x', 'y'])
        if get_z_func is None: return None
        return lambda xy_inputs: np.stack((xy_inputs[:,0], xy_inputs[:,1], np.array([get_z_func(x, y) for x, y in zip(xy_inputs[:,0], xy_inputs[:,1])], dtype=np.float64)), axis=-1)
    elif surface_type == 'parametric':
        get_x_func = _create_callable_from_code_string_sandbox(llm_code_str, 'get_x', ['u', 'v'])
        get_y_func = _create_callable_from_code_string_sandbox(llm_code_str, 'get_y', ['u', 'v'])
        get_z_func = _create_callable_from_code_string_sandbox(llm_code_str, 'get_z', ['u', 'v'])
        if not all([get_x_func, get_y_func, get_z_func]): return None
        return lambda uv_inputs: np.stack((np.array([get_x_func(u,v) for u,v in zip(uv_inputs[:,0],uv_inputs[:,1])], dtype=np.float64), 
                                          np.array([get_y_func(u,v) for u,v in zip(uv_inputs[:,0],uv_inputs[:,1])], dtype=np.float64), 
                                          np.array([get_z_func(u,v) for u,v in zip(uv_inputs[:,0],uv_inputs[:,1])], dtype=np.float64)), axis=-1)
    elif surface_type == 'implicit':
        get_f_func = _create_callable_from_code_string_sandbox(llm_code_str, 'get_f', ['x', 'y', 'z'])
        return get_f_func 
    return None

def _generate_points_from_callable_sandbox(surface_callable, surface_type, num_points, param_ranges):
    if surface_callable is None: return None
    num_samples_per_dim = int(np.sqrt(num_points)) 
    
    if surface_type == 'explicit':
        x_vals = np.linspace(param_ranges['x'][0], param_ranges['x'][1], num_samples_per_dim)
        y_vals = np.linspace(param_ranges['y'][0], param_ranges['y'][1], num_samples_per_dim)
        X, Y = np.meshgrid(x_vals, y_vals)
        input_coords = np.stack((X.flatten(), Y.flatten()), axis=-1)
        try:
            points = surface_callable(input_coords)
            if not isinstance(points, np.ndarray) or points.shape[1] != 3: raise ValueError("Invalid output shape")
            return points
        except Exception as e:
            
            return None
            
    elif surface_type == 'parametric':
        u_vals = np.linspace(param_ranges['u'][0], param_ranges['u'][1], num_samples_per_dim)
        v_vals = np.linspace(param_ranges['v'][0], param_ranges['v'][1], num_samples_per_dim)
        U, V = np.meshgrid(u_vals, v_vals)
        input_coords = np.stack((U.flatten(), V.flatten()), axis=-1)
        try:
            points = surface_callable(input_coords)
            if not isinstance(points, np.ndarray) or points.shape[1] != 3: raise ValueError("Invalid output shape")
            return points
        except Exception as e:
            
            return None

    elif surface_type == 'implicit':
        grid_dim = int(num_points**(1/3.0)) + 2
        x_vals = np.linspace(param_ranges['x'][0], param_ranges['x'][1], grid_dim)
        y_vals = np.linspace(param_ranges['y'][0], param_ranges['y'][1], grid_dim)
        z_vals = np.linspace(param_ranges['z'][0], param_ranges['z'][1], grid_dim)
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        all_volume_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
        try:
            f_values = np.array([surface_callable(p[0], p[1], p[2]) for p in all_volume_points], dtype=np.float64)
            on_surface_mask = np.isclose(f_values, 0.0, atol=1e-3)
            points = all_volume_points[on_surface_mask]
            if len(points) > num_points: points = points[np.random.choice(len(points), num_points, replace=False)]
            return points if points.shape[0] > 0 else None
        except Exception as e:
            
            return None
    return None

def _compute_chamfer_distance_sandbox(pc1, pc2):
    if pc1 is None or pc2 is None or pc1.shape[0] == 0 or pc2.shape[0] == 0: return float('inf')
    from scipy.spatial.distance import cdist 
    distances1_sq = np.min(cdist(pc1, pc2, metric='euclidean')**2, axis=1)
    distances2_sq = np.min(cdist(pc2, pc1, metric='euclidean')**2, axis=1)
    return np.mean(distances1_sq) + np.mean(distances2_sq)
"""


eval_spec = """
import numpy as np
import math
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

{}

@evaluate.run
def evaluate(data_context: dict) -> float:
    \"\"\"
    Evaluates a generated 3D surface equation.
    data_context contains 'gt_samples_xyz', 'surface_type', 'param_ranges_for_gen', 'llm_code_str'.
    Returns a scalar score (negative Chamfer Distance).
    \"\"\"
    gt_samples_xyz = data_context['gt_samples_xyz']
    surface_type = data_context['surface_type']
    param_ranges_for_gen = data_context['param_ranges_for_gen']
    llm_code_str = data_context['llm_code_str'] 
    
    predicted_callable = _get_surface_callable_sandbox(llm_code_str, surface_type)
    if predicted_callable is None:
        return -float('inf') 

    predicted_pc = _generate_points_from_callable_sandbox(
        predicted_callable,
        surface_type,
        num_points={}, 
        param_ranges=param_ranges_for_gen
    )
    
    if predicted_pc is None or predicted_pc.shape[0] == 0:
        return -float('inf') 

    chamfer_dist = _compute_chamfer_distance_sandbox(predicted_pc, gt_samples_xyz)
    
    if np.isnan(chamfer_dist) or np.isinf(chamfer_dist):
        return -float('inf')
    else:
        return -chamfer_dist
""".format(_INLINED_UTILS_FOR_SANDBOX, _NUM_POINTS_FOR_SANDBOX_PC)


optimization_spec = """
import numpy as np
def equation(data_context: dict) -> list:
    \"\"\"
    Placeholder optimization function.
    Returns dummy parameters or a representation of the discovered symbolic parameters.
    \"\"\"
    return [1.0, 1.0, 1.0] 
@evaluate.run
def optimize(data_context: dict) -> list:
    return equation(data_context)
"""

execution_spec = """
import numpy as np
def equation(data_context: dict) -> list:
    \"\"\"
    Placeholder execution function.
    Returns dummy output as actual point generation is done via callable from bench.utils.
    \"\"\"
    return [0.0, 0.0, 0.0] 
@evaluate.run
def execute(data_context: dict) -> list:
    return equation(data_context)
"""

# def problem_from_task(task: SEDTask, problem: Problem) -> Problem:
#     """
#     Reconstructs a Problem object from an SEDTask and the original Problem object.
#     This is necessary because the LLMSR pipeline's `discover` method
#     now needs to access the full Problem object's data (ID/OOD splits).
#     """
#     return Problem(
#         dataset_identifier=problem.dataset_identifier,
#         equation_idx=task.name,
#         gt_equation=problem.gt_equation,
#         samples={
#             'train_data': task.samples,
#             'id_test_data': problem.test_samples,
#             'ood_test_data': problem.ood_test_samples,
#         }
#     )
class LLMSRSearcher(BaseSearcher): 
    def __init__(self, 
                 name: str, 
                 cfg: config_lib.Config, 
                 SamplerClass: Type[sampler.LLM],
                 global_max_sample_num: int, 
                 log_path: str) -> None:
        super().__init__(name)
        self.cfg = cfg
        self.SamplerClass = SamplerClass 
        self.global_max_sample_num = global_max_sample_num
        self.log_path = log_path

        self.class_config = config_lib.ClassConfig(llm_class=self.SamplerClass, 
                                               sandbox_class=evaluator.LocalSandbox)

    def _create_llm_prompt(self, problem: Problem, surface_type: str) -> List[Dict[str, str]]:
        """
        Creates the LLM prompt based on the Problem object and surface type, using SURFACEBENCH templates.
        This prompt will instruct the LLM to generate `get_x/y/z` or `get_f` functions.
        """
        points_string_for_prompt = ""
        for i, point in enumerate(problem.train_samples):
            points_string_for_prompt += "({:.4f}, {:.4f}, {:.4f})".format(point[0], point[1], point[2])
            if i < problem.train_samples.shape[0] - 1:
                points_string_for_prompt += ", "
            if (i + 1) % 4 == 0 and i < problem.train_samples.shape[0] - 1:
                points_string_for_prompt += "\n"

        category_prop = [prop for prop in problem.gt_equation.symbol_properties if 'category:' in prop]
        category = category_prop[0].split(':')[1] if category_prop else "Unknown Category"
        
        input_vars_desc_str = ''.join(["{}: {}\n".format(s, d) for s, d in zip(problem.gt_equation.symbols, problem.gt_equation.symbol_descs)])
        
        user_prompt_template = ""
        
        type_info = _SURFACE_TYPE_INFO.get(surface_type)
        if not type_info:
            warnings.warn("Unsupported surface type '{}' for prompt generation.".format(surface_type))
            return []

        if surface_type == 'explicit':
            user_prompt_template_raw = """
            Category: {category}
            Input variables:
            {input_vars_desc_str}
            Output variable:
            z: height or surface value

            Sampled 3D data points from the surface (x, y, z):
            {points_string_for_prompt}

            Observations about the surface:
            {observations}

            Please provide the Python code for the function `get_z({input_symbol_0}, {input_symbol_1})` that describes this surface.
            Wrap your code in a Python markdown block, and remember to import the 'math' module.
            ```python
            import math

            # Provide your function here:
            # def get_z({input_symbol_0}, {input_symbol_1}):
            #    ...
            """
            user_prompt_template = user_prompt_template_raw.format(
                category=category,
                input_vars_desc_str=input_vars_desc_str,
                points_string_for_prompt=points_string_for_prompt,
                observations=problem.gt_equation.desc if problem.gt_equation.desc else "No specific observations provided.",
                input_symbol_0=type_info['input_symbols_for_llm_code'][0],
                input_symbol_1=type_info['input_symbols_for_llm_code'][1]
                )
        elif surface_type == 'parametric':
            user_prompt_template_raw = """
                Category: {category}
                Input variables:
                {input_vars_desc_str}
                Output variables:
                x(u,v), y(u,v), z(u,v)

                Sampled 3D data points from the surface (x, y, z):
                {points_string_for_prompt}

                Observations about the surface:
                {observations}

                Please provide the Python code for the functions get_x({input_symbol_0}, {input_symbol_1}), get_y({input_symbol_0}, {input_symbol_1}), and get_z({input_symbol_0}, {input_symbol_1}) that describe this surface.
                ```Python
                import math
                import math
                # Provide your functions here:
                # def get_x(u, v):
                #    ...
                # def get_y(u, v):
                #    ...
                # def get_z(u, v):
                #    ...
                """
            user_prompt_template = user_prompt_template_raw.format(
                category=category,
                input_vars_desc_str=input_vars_desc_str,
                points_string_for_prompt=points_string_for_prompt,
                observations=problem.gt_equation.desc if problem.gt_equation.desc else "No specific observations provided.",
                input_symbol_0=type_info['input_symbols_for_llm_code'][0],
                input_symbol_1=type_info['input_symbols_for_llm_code'][1]
                )
            
        elif surface_type == 'implicit':
            user_prompt_template_raw = """
            Category: {category}

            Input variables:
            {input_vars_desc_str}
            Output:
            f(x, y, z): a symbolic function whose zero level set defines the surface

            Sampled 3D data points from the surface (x, y, z):
            {points_string_for_prompt}

            Observations about the surface:
            {observations}

            Please propose a symbolic expression f(x, y, z) that defines this surface.
            Provide the Python code for the function get_f({input_symbol_0}, {input_symbol_1}, {input_symbol_2}) that describes this surface.
```python
import math

# Provide your function here:
# def get_f(x, y, z):
#    ...
"""         
            user_prompt_template = user_prompt_template_raw.format(
                category=category,
                input_vars_desc_str=input_vars_desc_str,
                points_string_for_prompt=points_string_for_prompt,
                observations=problem.gt_equation.desc if problem.gt_equation.desc else "No specific observations provided.",
                input_symbol_0=type_info['input_symbols_for_llm_code'][0],
                input_symbol_1=type_info['input_symbols_for_llm_code'][1],
                input_symbol_2=type_info['input_symbols_for_llm_code'][2]
                )
        else:
            warnings.warn("Unsupported surface type '{}' for prompt generation.".format(surface_type))
            return []
        
        return [{"role": "user", "content": user_prompt_template}]

    def _extract_llm_code_block(self, llm_raw_output: str) -> Optional[str]:
        """Extracts the Python code block from LLM's raw output."""
        code_block_match = re.search(r"```python\s*(.*?)\s*```", llm_raw_output, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()
        else:
            warnings.warn("Could not find Python code block in LLM output.")
            return None

    def discover(self, task: SEDTask) -> List[SearchResult]:
            """
            Discovers a symbolic equation for the given problem using the LLMSR method.
            This LLMSR implementation runs the pipeline.main to execute the LLMSR loop.
            """
            search_results = []
            
            # We get the problem metadata and data from the SEDTask object
            # problem = problem_from_task(task,Problem)

            surface_type = "unknown"
            for prop in task.gt_equation.symbol_properties:
                if prop.startswith("surface_type:"):
                    surface_type = prop.split(":")[1]
                    break
            
            if surface_type == "unknown":
                warnings.warn(f"Surface type not found for problem {task.equation_idx}. Skipping discovery.")
                return []

            evaluation_contexts_for_evaluator = []
            
            if task.test_samples is not None and task.test_samples.size > 0:
                evaluation_contexts_for_evaluator.append({
                    'test_set_type': 'ID',
                    'gt_samples_xyz': task.test_samples, 
                    'surface_type': surface_type,
                    'param_ranges_for_gen': _DEFAULT_EVAL_RANGES[surface_type],
                    'llm_code_str': "" 
                })
            
            if task.ood_test_samples is not None and task.ood_test_samples.size > 0:
                evaluation_contexts_for_evaluator.append({
                    'test_set_type': 'OOD',
                    'gt_samples_xyz': task.ood_test_samples,
                    'surface_type': surface_type,
                    'param_ranges_for_gen': _DEFAULT_EVAL_RANGES[surface_type],
                    'llm_code_str': "" 
                })
            
            if not evaluation_contexts_for_evaluator:
                warnings.warn(f"No evaluation samples (ID or OOD) for problem {task.equation_idx}. Skipping discovery.")
                return []

            type_info = _SURFACE_TYPE_INFO.get(surface_type)
            if not type_info:
                warnings.warn(f"Unsupported surface type '{surface_type}' for equation specification. Using dummy.")
                equation_specification_template = """
                    import numpy as np
                    import math
                    @equation.evolve
                    def equation(x: np.ndarray) -> np.ndarray:
                        return np.zeros((x.shape[0], 3))
                    """
            else:
                input_args_str_for_eq_func = type_info['input_args_str']
                return_type_hint_for_eq_func = type_info['output_return_type']

                args_docstring_raw = ''.join([
                    "{}: A numpy array representing observations of {}.\n".format(s, d)
                    for s, d in zip(task.gt_equation.symbols, task.gt_equation.symbol_descs)
                ])
                args_docstring_for_eq_func = textwrap.indent(args_docstring_raw.strip(), " " * 8)
                return_docstring_for_eq_func = textwrap.indent(args_docstring_raw.strip(), " " * 8)

                equation_specification_template = """\
                
import numpy as np
import math
from typing import List, Tuple 

@equation.evolve 
def equation({input_args_str_for_eq_func}) -> {return_type_hint_for_eq_func}:
    \"\"\" Mathematical function to discover a 3D surface.
    
    Args:
{args_docstring_for_eq_func}    Return:
        {return_docstring_for_eq_func}
    \"\"\"
    pass 
""".format(
                input_args_str_for_eq_func=input_args_str_for_eq_func,
                return_type_hint_for_eq_func=return_type_hint_for_eq_func,
                args_docstring_for_eq_func=args_docstring_for_eq_func,
                return_docstring_for_eq_func=return_docstring_for_eq_func
            )
                equation_specification_template = textwrap.dedent(equation_specification_template)
                sampler.Sampler._global_samples_nums = 1

            profiler = pipeline.main(
                specification=eval_spec + "\n\n" + equation_specification_template, 
                inputs=evaluation_contexts_for_evaluator, 
                config=self.cfg,
                max_sample_nums=self.global_max_sample_num,
                class_config=self.class_config,
                log_dir=os.path.join(self.log_path, task.equation_idx), 
            )

            best_program_str = profiler._cur_best_program_str
            
            program_obj_from_best = code_manipulation.text_to_program(best_program_str)
            evolved_equation_func = program_obj_from_best.get_function("equation")
            
            llm_generated_surface_code = evolved_equation_func.body 
            
            discovered_surface_callable = get_surface_callable_from_llm_code(llm_generated_surface_code, surface_type)

            best_equation = Equation(
                symbols=task.gt_equation.symbols, 
                symbol_descs=task.gt_equation.symbol_descs,
                symbol_properties=task.gt_equation.symbol_properties,
                expression=best_program_str, 
                program_format=llm_generated_surface_code,
                lambda_format=discovered_surface_callable
            )

            return [
                SearchResult(
                    equation=best_equation,
                    aux={
                        "best_program_sample_order": profiler._cur_best_program_sample_order, 
                        "best_program_score": profiler._cur_best_program_score, 
                        "surface_type": surface_type, 
                        "llm_raw_output_code": llm_generated_surface_code
                    },
                )
            ]

    def program_to_function(self, task: SEDTask, best_program_str: str) -> Equation:
        surface_type = "unknown"
        for prop in task.gt_equation.symbol_properties:
            if prop.startswith("surface_type:"):
                surface_type = prop.split(":")[1]
                break
        if surface_type == "unknown":
            warnings.warn(
                f"Surface type not found in symbol_properties for problem {task.equation_idx}. Cannot convert program to function.")
            return Equation(symbols=[], symbol_descs=[], symbol_properties=[], expression="Error", program_format="", lambda_format=None)

        program_obj_from_str = code_manipulation.text_to_program(best_program_str)
        evolved_equation_func = program_obj_from_str.get_function("equation")
        llm_generated_surface_code = evolved_equation_func.body

        predicted_callable = None
        try:
            predicted_callable = get_surface_callable_from_llm_code(llm_generated_surface_code, surface_type)
        except Exception as e:
            warnings.warn(f"Error creating callable from program_to_function for {task.equation_idx}: {e}")

        return Equation(
            symbols=task.gt_equation.symbols,
            symbol_descs=task.gt_equation.symbol_descs,
            symbol_properties=task.gt_equation.symbol_properties,
            expression=best_program_str,
            program_format=llm_generated_surface_code,
            lambda_format=predicted_callable
        )
