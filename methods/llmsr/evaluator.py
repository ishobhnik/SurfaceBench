from abc import abstractmethod, ABC
import ast
import time
from collections.abc import Sequence
from functools import reduce
import copy
import math
import numpy as np
from typing import Any, Type, Dict, List, Optional, Tuple
import multiprocessing
import ctypes
import numpy as np
import warnings # Needed for warnings module
from methods.llmsr import profile

# Local imports from llmsr package
import methods.llmsr.code_manipulation as code_manipulation
from methods.llmsr import buffer
from methods.llmsr import evaluator_accelerate
import sys
import os
bench_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bench'))
if bench_dir_path not in sys.path:
    sys.path.insert(0, bench_dir_path)

from bench.utils import ( # Import Chamfer, Hausdorff, and point generation utilities
    get_surface_callable_from_llm_code,
    generate_points_from_surface_callable,
    compute_chamfer_distance,
    compute_hausdorff_distance
)

class _FunctionLineVisitor(ast.NodeVisitor):
    """ Visitor that finds the last line number of a function with a given name."""
    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None: 
        """ Collect the end line number of the target function."""
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        """ Line number of the final line of function `target_function_name`."""
        assert self._function_end_line is not None 
        return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
    """ Extract the body of the generated function, trimming anything after it.
    Please note that the indentation is REQUIRED !!!
    """
    if not generated_code:
        return ''

    code = f'def fake_function_header():\n{generated_code}'

    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            if e.lineno is None:
                return ''
            code = '\n'.join(code.splitlines()[:e.lineno - 1])

    if not code:
        return ''

    visitor = _FunctionLineVisitor('fake_function_header')
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    return '\n'.join(body_lines) + '\n\n'


def _sample_to_program(
        generated_code: str,
        version_generated: int | None,
        template: code_manipulation.Program,
        function_to_evolve: str,
) -> Tuple[code_manipulation.Function, str]:
    """ 
    Return the compiled generated function and the full runnable program.
    This function replaces the body of the `function_to_evolve` in the `template`.
    """
    body = _trim_function_body(generated_code)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            code=body,
            source_name=f'{function_to_evolve}_v{version_generated}',
            target_name=function_to_evolve
        )

    program_copy = copy.deepcopy(template)
    evolved_function = program_copy.get_function(function_to_evolve)
    evolved_function.body = body
    
    return evolved_function, str(program_copy)


class Sandbox(ABC):
    """ Sandbox for executing generated code. """
    @abstractmethod
    def run(
            self,
            program: str,
            function_to_run: str,
            function_to_evolve: str,
            inputs_for_equation: Any,
            surface_type: str,
            problem_params: Dict[str, Any],
            timeout_seconds: int,
            **kwargs
    ) -> Tuple[Any, bool]:
        """ Return `function_to_run(...)`'s result and whether execution succeeded. """
        raise NotImplementedError(
            'Must provide a sandbox for executing untrusted code.')


class LocalSandbox(Sandbox):
    """
    Secure environment for executing and evaluating LLM generated programs locally using multiprocessing.
    """
    def __init__(self, verbose=False, numba_accelerate=False):
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate


    def run(self, program: str, function_to_run: str, function_to_evolve: str, 
            inputs_for_equation: List[np.ndarray],
            surface_type: str, 
            problem_params: Dict[str, Any],
            timeout_seconds: int,
            get_result_sleep_time=0.1, result_array_shape=None,  **kwargs) -> Tuple[Any, bool]:
        """
        Execute the given program sample in a subprocess and return its results.
        
        Args:
            program: The full Python program string to execute.
            function_to_run: The name of the function in 'program' to call (e.g., 'execute', 'evaluate').
            function_to_evolve: The name of the function being evolved (e.g., 'equation').
            inputs_for_equation: The input data for `function_to_run` (e.g., [x_arr, y_arr]).
            surface_type: Type of surface ('explicit', 'parametric', 'implicit').
            problem_params: Dictionary with problem-specific parameters (e.g., gt_samples_xyz, etc.).
            timeout_seconds: Timeout for execution.
            result_array_shape: Expected shape of the result from `function_to_run` (e.g., (N,3) for point clouds or (N,) for scalar/f-values).
        """
        
        # Use shared memory for large NumPy arrays between processes
        array_size = reduce(lambda x,y: x*y, result_array_shape, 1) if result_array_shape else 0
        shared_array_np = None
        if array_size > 0:
            shared_array_base = multiprocessing.RawArray(ctypes.c_double, array_size)
            shared_array_np = np.ndarray(result_array_shape, dtype=np.float64, buffer=shared_array_base)

        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, function_to_run, function_to_evolve, inputs_for_equation, 
                  surface_type, problem_params,
                  self._numba_accelerate, result_queue, shared_array_np)
        )
        process.start()
        process.join(timeout=timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join()
            results = None
            runs_ok = False
            warnings.warn(f"Program execution timed out after {timeout_seconds}s for {function_to_run}.")
        else:
            status_from_queue, runs_ok = self._get_results(result_queue, sleep_time=get_result_sleep_time)
            if not runs_ok:
                results = None
                if status_from_queue:
                    warnings.warn(f"Program execution failed for {function_to_run}: {status_from_queue}")
            else:
                if shared_array_np is not None:
                    results = shared_array_np
                else:
                    results = status_from_queue

        if self._verbose:
            self._print_evaluation_details(program, (results, runs_ok), **kwargs)

        return results, runs_ok


    def _get_results(self, queue, sleep_time=0.1):
        for _ in range(int(5 / sleep_time)):
            if not queue.empty():
                return queue.get_nowait()
            time.sleep(sleep_time)
        return None, False

    def _print_evaluation_details(self, program, results, **kwargs):
        print('================= Evaluated Program =================')
        function = code_manipulation.text_to_program(program).get_function(kwargs.get('func_to_evolve', 'equation'))
        print(f'{str(function).strip()}\n-----------------------------------------------------')
        print(f'Score: {results}\n=====================================================\n\n')

    def _compile_and_run_function(self, program: str, function_to_run: str, function_to_evolve: str, 
                                  dataset_inputs_for_equation: List[np.ndarray],
                                  surface_type: str, problem_params: Dict[str, Any],
                                  numba_accelerate: bool, result_queue: multiprocessing.Queue, 
                                  shared_array_np: Optional[np.ndarray]):
        """
        This function runs in a separate process. It executes the program.
        It evaluates the LLM's `equation` function and prepares output for shared memory.
        """
        try:
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program,
                    function_to_evolve=function_to_evolve
                )
            
            all_globals_namespace = {
                "np": np, "math": math,
                "sin": math.sin, "cos": math.cos, "exp": math.exp, "log": math.log,
                "sqrt": math.sqrt, "abs": abs, "fabs": math.fabs, "pi": math.pi,
                "problem_params": problem_params,
            }
            for name in dir(math):
                if not name.startswith('__'):
                    all_globals_namespace[name] = getattr(math, name)

            exec(program, all_globals_namespace)
            
            function_to_run_callable = all_globals_namespace[function_to_run]
            raw_results = function_to_run_callable(*dataset_inputs_for_equation)

            processed_results = None
            if surface_type == 'explicit':
                if isinstance(raw_results, np.ndarray) and raw_results.ndim == 1:
                    if len(dataset_inputs_for_equation) == 2 and dataset_inputs_for_equation[0].shape == raw_results.shape:
                        processed_results = np.stack((dataset_inputs_for_equation[0], dataset_inputs_for_equation[1], raw_results), axis=-1)
                    else:
                        warnings.warn("Explicit surface output shape mismatch or inputs missing.")
                else:
                    warnings.warn("Explicit surface did not return 1D numpy array.")
            elif surface_type == 'parametric':
                if isinstance(raw_results, tuple) and all(isinstance(r, np.ndarray) for r in raw_results):
                    processed_results = np.stack(raw_results, axis=-1)
                else:
                    warnings.warn("Parametric surface did not return tuple of numpy arrays.")
            elif surface_type == 'implicit':
                if isinstance(raw_results, np.ndarray) and raw_results.ndim == 1:
                    processed_results = raw_results
                else:
                    warnings.warn("Implicit surface did not return 1D numpy array for f-values.")
            else:
                warnings.warn(f"Unknown surface type {surface_type} in sandbox processing.")
            
            if processed_results is None:
                result_queue.put((None, False))
                return

            if shared_array_np is not None:
                if processed_results.shape == shared_array_np.shape:
                    shared_array_np[:] = processed_results
                    result_queue.put((None, True))
                else:
                    warnings.warn(f"Processed result shape {processed_results.shape} does not match shared array shape {shared_array_np.shape}. Required: {shared_array_np.shape}")
                    result_queue.put((None, False))
            else:
                warnings.warn("Shared array not provided for 3D surface results. Results might not be transferred correctly.")
                result_queue.put((processed_results, True))
            
        except Exception as e:
            result_queue.put((str(e), False))

def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    """ Return whether the generated function is calling an earlier version. """
    for name in code_manipulation.get_functions_called(program):
        if name.startswith(f'{function_to_evolve}_v'):
            return True
    return False



class Evaluator:
    """ Class that analyses functions generated by LLMs. """

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            template: code_manipulation.Program,
            function_to_evolve: str, 
            function_to_run: str, 
            inputs: Sequence[Any], 
            timeout_seconds: int = 30,
            sandbox_class: Type[Sandbox] = Sandbox
    ):
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox_class()

    def analyse(
            self,
            sample: str,
            island_id: int | None,
            version_generated: int | None,
            **kwargs 
    ) -> None:
        """ Compile the hypothesis sample into a program and executes it on test inputs. """
        new_function, program = _sample_to_program(
            sample, version_generated, self._template, self._function_to_evolve)
        scores_per_test = {}

        time_reset = time.time()
        
        for current_input in self._inputs:
            test_output, runs_ok = self._sandbox.run(
                program=program,
                function_to_run=self._function_to_run,
                function_to_evolve=self._function_to_evolve,
                inputs_for_equation=current_input['gt_samples_xyz'],
                surface_type=current_input['surface_type'],
                problem_params=current_input,
                timeout_seconds=self._timeout_seconds,
                result_array_shape=current_input.get('result_array_shape', (current_input['gt_samples_xyz'].shape[0], 3))
            )
            if runs_ok and not _calls_ancestor(program, self._function_to_evolve) and test_output is not None:
                if not isinstance(test_output, (int, float)):
                    print(f'Error: test_output is {test_output}')
                    raise ValueError('@function.run did not return an int/float score.')
                scores_per_test[current_input] = test_output

        evaluate_time = time.time() - time_reset
        # print("scores_per_test", scores_per_test)
        if scores_per_test:
            # print("*")
            self._database.register_program(
                new_function,
                island_id,
                scores_per_test,
                **kwargs,
                evaluate_time=evaluate_time
            )
        
        else:
            profiler: profile.Profiler = kwargs.get('profiler', None)
            if profiler:
                global_sample_nums = kwargs.get('global_sample_nums', None)
                sample_time = kwargs.get('sample_time', None)
                new_function.global_sample_nums = global_sample_nums
                new_function.score = None
                new_function.sample_time = sample_time
                new_function.evaluate_time = evaluate_time
                profiler.register_function(new_function)