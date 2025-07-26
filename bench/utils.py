import numpy as np
import sympy
import math
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree 
import warnings
from typing import List, Optional, Callable, Dict, Any

_MATH_GLOBALS = {
    "math": math,
    "np": np,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "log": math.log, "log10": math.log10,
    "sqrt": math.sqrt, "abs": abs, "fabs": math.fabs,
    "pow": math.pow,
    "pi": math.pi, "e": math.e,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "asinh": math.asinh, "acosh": math.acosh, "atanh": math.atanh,
    "inf": float('inf'), "nan": float('nan')
}
_SYMPY_LOCALS = {
    "x": sympy.Symbol('x'), "y": sympy.Symbol('y'), "z": sympy.Symbol('z'),
    "u": sympy.Symbol('u'), "v": sympy.Symbol('v'),
    "t": sympy.Symbol('t'), 
    "R": sympy.Symbol('R'), 
    "pi": sympy.pi,
    "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
    "exp": sympy.exp, "log": sympy.log, "sqrt": sympy.sqrt, "Abs": sympy.Abs,
    "Pow": sympy.Pow 
}


def strexpression2sympy(eq_text: str, locals_dict: Optional[Dict[str, Any]] = None) -> Optional[sympy.Expr]:
    """Converts a string expression to a SymPy expression."""
    if locals_dict is None:
        locals_dict = _SYMPY_LOCALS

    eq_text = eq_text.replace("^", "**") 
    eq_text = eq_text.replace("π", "pi") 
    eq_text = eq_text.replace("abs(", "Abs(") 
    eq_text = eq_text.replace("log(", "log(")

    try:
        eq_sympy = sympy.sympify(eq_text, locals=locals_dict, evaluate=False)
        return eq_sympy
    except (sympy.SympifyError, SyntaxError) as e:
        warnings.warn(f"SymPy parsing failed for expression '{eq_text}': {e}")
        return None

def create_callable_from_code_string(code_str: str, func_name: str, input_vars: List[str]) -> Optional[Callable]:
    """
    Executes a string of Python code and extracts a specific callable function.
    Provides a restricted environment.
    """
    local_env = {}
    try:
        exec(code_str, _MATH_GLOBALS, local_env)
        
        func = local_env.get(func_name)
        if func and callable(func):
            return func
        else:
            warnings.warn(f"Function '{func_name}' not found or not callable in LLM output.")
            return None
    except Exception as e:
        warnings.warn(f"Error executing LLM-generated code to get '{func_name}': {e}")
        return None

def get_surface_callable_from_llm_code(llm_code_str: str, surface_type: str) -> Optional[Callable]:
    """
    Extracts and compiles a single callable Python function from LLM-generated code string
    that can generate (x,y,z) points given appropriate inputs (x,y or u,v).
    """
    if surface_type == 'explicit':
        get_z_func = create_callable_from_code_string(llm_code_str, 'get_z', ['x', 'y'])
        if get_z_func is None: return None
        
        def explicit_surface_callable(xy_inputs: np.ndarray) -> np.ndarray:
            x_vals = xy_inputs[:, 0]
            y_vals = xy_inputs[:, 1]
            try:
                z_vals = np.array([get_z_func(x_val, y_val) for x_val, y_val in zip(x_vals, y_vals)])
                return np.stack((x_vals, y_vals, z_vals), axis=-1)
            except Exception as e:
                warnings.warn(f"Error evaluating explicit surface callable: {e}")
                return np.full((xy_inputs.shape[0], 3), np.nan) # Return NaN array on error
        return explicit_surface_callable
    
    elif surface_type == 'parametric':
        get_x_func = create_callable_from_code_string(llm_code_str, 'get_x', ['u', 'v'])
        get_y_func = create_callable_from_code_string(llm_code_str, 'get_y', ['u', 'v'])
        get_z_func = create_callable_from_code_string(llm_code_str, 'get_z', ['u', 'v'])
        if not all([get_x_func, get_y_func, get_z_func]): return None
        
        def parametric_surface_callable(uv_inputs: np.ndarray) -> np.ndarray:
            u_vals = uv_inputs[:, 0]
            v_vals = uv_inputs[:, 1]
            try:
                x_vals = np.array([get_x_func(u, v) for u, v in zip(u_vals, v_vals)])
                y_vals = np.array([get_y_func(u, v) for u, v in zip(u_vals, v_vals)])
                z_vals = np.array([get_z_func(u, v) for u, v in zip(u_vals, v_vals)])
                return np.stack((x_vals, y_vals, z_vals), axis=-1)
            except Exception as e:
                warnings.warn(f"Error evaluating parametric surface callable: {e}")
                return np.full((uv_inputs.shape[0], 3), np.nan) 
        return parametric_surface_callable
    
    elif surface_type == 'implicit':
        # For implicit surfaces, this callable will typically evaluate f(x,y,z) for a given point.
        get_f_func = create_callable_from_code_string(llm_code_str, 'get_f', ['x', 'y', 'z'])
        if get_f_func is None: return None
        return get_f_func
    
    else:
        warnings.warn(f"Unsupported surface type for callable extraction: {surface_type}")
        return None

def generate_points_from_surface_callable(
    surface_callable: Callable,
    surface_type: str,
    num_points: int = 5000,
    param_ranges: Optional[Dict[str, List[float]]] = None,
) -> Optional[np.ndarray]:
    """
    Generates a uniform 3D point cloud from a given callable surface function.
    For implicit, it provides a very basic sampling, a more robust solution
    would involve marching cubes or specialized meshing algorithms.
    """
    if surface_callable is None:
        return None

    num_samples_per_dim = int(np.sqrt(num_points))

    if surface_type == 'explicit':
        if param_ranges is None or 'x' not in param_ranges or 'y' not in param_ranges:
            warnings.warn("Using default ranges for explicit surface (x,y).")
            x_range, y_range = (-5, 5), (-5, 5)
        else:
            x_range, y_range = param_ranges['x'], param_ranges['y']
            
        x_vals = np.linspace(x_range[0], x_range[1], num_samples_per_dim)
        y_vals = np.linspace(y_range[0], y_range[1], num_samples_per_dim)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        input_coords = np.stack((X.flatten(), Y.flatten()), axis=-1)
        points = surface_callable(input_coords) # Returns (N, 3) for explicit
        
    elif surface_type == 'parametric':
        if param_ranges is None or 'u' not in param_ranges or 'v' not in param_ranges:
            warnings.warn("Using default ranges for parametric surface (u,v).")
            u_range, v_range = (0, 2 * np.pi), (0, np.pi)
        else:
            u_range, v_range = param_ranges['u'], param_ranges['v']

        u_vals = np.linspace(u_range[0], u_range[1], num_samples_per_dim)
        v_vals = np.linspace(v_range[0], v_range[1], num_samples_per_dim)
        U, V = np.meshgrid(u_vals, v_vals)
        
        input_coords = np.stack((U.flatten(), V.flatten()), axis=-1)
        points = surface_callable(input_coords) # Returns (N, 3) for parametric

    elif surface_type == 'implicit':
        warnings.warn("Basic point cloud generation for implicit functions. Consider specialized meshing for rigor.")
        if param_ranges is None or 'x' not in param_ranges or 'y' not in param_ranges or 'z' not in param_ranges:
            warnings.warn("Using default ranges for implicit surface (x,y,z).")
            x_range, y_range, z_range = (-5, 5), (-5, 5), (-5, 5)
        else:
            x_range, y_range, z_range = param_ranges['x'], param_ranges['y'], param_ranges['z']

        grid_dim = int(num_points**(1/3.0)) + 2 # Adjust for cubic sampling
        x_vals = np.linspace(x_range[0], x_range[1], grid_dim)
        y_vals = np.linspace(y_range[0], y_range[1], grid_dim)
        z_vals = np.linspace(z_range[0], z_range[1], grid_dim)
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        
        all_volume_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
        
        f_values = np.array([surface_callable(p[0], p[1], p[2]) for p in all_volume_points])
        
        on_surface_mask = np.isclose(f_values, 0.0, atol=1e-3) 
        points = all_volume_points[on_surface_mask]

        if len(points) == 0 and num_points > 0:
            warnings.warn("No points found on implicit surface with current sampling and tolerance.")
            return None
        
        if len(points) > num_points:
            points = points[np.random.choice(len(points), num_points, replace=False)]

    else:
        warnings.warn(f"Unsupported surface type: {surface_type} for point generation.")
        return None

    return points if points.shape[0] > 0 else None


def compute_chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """
    Computes the Chamfer Distance between two point clouds.
    CD(A, B) = mean_a (min_b ||a-b||^2) + mean_b (min_a ||b-a||^2)
    Uses mean of squared distances for a balanced metric.
    Args:
        pc1 (np.ndarray): First point cloud (N1, 3).
        pc2 (np.ndarray): Second point cloud (N2, 3).
    Returns:
        float: Chamfer distance. Returns infinity if any point cloud is empty.
    """
    if pc1 is None or pc2 is None or pc1.shape[0] == 0 or pc2.shape[0] == 0:
        return float('inf')

    # Calculate squared Euclidean distances from each point in pc1 to its nearest in pc2
    distances1_sq = np.min(cdist(pc1, pc2, metric='euclidean')**2, axis=1)
    
    # Calculate squared Euclidean distances from each point in pc2 to its nearest in pc1
    distances2_sq = np.min(cdist(pc2, pc1, metric='euclidean')**2, axis=1)
    
    # Chamfer distance is the sum of the means of these squared minimum distances
    return np.mean(distances1_sq) + np.mean(distances2_sq)

def compute_hausdorff_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """
    Computes the Hausdorff Distance between two point clouds.
    H(A, B) = max(h(A, B), h(B, A)) where h(A, B) = max_a (min_b ||a-b||)
    Args:
        pc1 (np.ndarray): First point cloud (N1, 3).
        pc2 (np.ndarray): Second point cloud (N2, 3).
    Returns:
        float: Hausdorff distance. Returns infinity if any point cloud is empty.
    """
    if pc1 is None or pc2 is None or pc1.shape[0] == 0 or pc2.shape[0] == 0:
        return float('inf')

    # Build KD-Trees for efficient nearest neighbor search (more common for Hausdorff)
    tree1 = KDTree(pc1)
    tree2 = KDTree(pc2)

    # h(pc1, pc2) = max_a (min_b ||a-b||)
    distances1, _ = tree1.query(pc2)
    h_pc1_pc2 = np.max(distances1) 

    # h(pc2, pc1) = max_b (min_a ||b-a||)
    distances2, _ = tree2.query(pc1) 
    h_pc2_pc1 = np.max(distances2)

    return max(h_pc1_pc2, h_pc2_pc1)

def evaluate_expression(expression: str, symbols: List[str], input_values: np.ndarray):
    '''
    Args:
        expression (str): equation in str format
        symbols (list): names of input variables (including output, if it's the first)
        input_values (ndarray): a numpy array whose shape is (num data point x num input variables - 1)
    '''
    expression = expression.replace("^", "**")
    
    input_syms_str = ','.join(symbols[1:])

    try:
        _exec_globals = {"math": math, "np": np}
        lambda_code = f"lambda {input_syms_str}: {expression}"
        exp_as_func = eval(lambda_code, _exec_globals)
    except Exception as e:
        warnings.warn(f"Error creating scalar lambda for '{expression}': {e}")
        return np.full(input_values.shape[0], np.nan) 

    Y = []
    for i in range(len(input_values)):
        try:
            Y.append(exp_as_func(*list(input_values[i])))
        except Exception as e:
            warnings.warn(f"Error evaluating scalar expression at point {input_values[i]}: {e}")
            Y.append(np.nan)
    
    return np.array(Y, dtype=np.float64)

# if __name__ == "__main__":
#     print("--- Testing utils.py surface evaluation functions ---")

#     # Example 1: Explicit Surface (z = x^2 + y^2)
#     print("\n--- Testing Explicit Surface (z = x^2 + y^2) ---")
    
#     # Define a simple explicit callable directly for testing purposes
#     # In reality, this would come from LLM output parsed by get_surface_callable_from_llm_code
#     def gt_z_explicit(x, y):
#         return x**2 + y**2
    
#     def llm_z_explicit(x, y): # Slightly off for testing error
#         return 0.9*x**2 + 1.1*y**2 + 0.1

#     # Wrapper callables to match get_surface_callable_from_llm_code output format
#     def wrapped_gt_explicit_callable(xy_inputs):
#         x_vals = xy_inputs[:, 0]
#         y_vals = xy_inputs[:, 1]
#         z_vals = np.array([gt_z_explicit(x, y) for x, y in zip(x_vals, y_vals)])
#         return np.stack((x_vals, y_vals, z_vals), axis=-1)

#     def wrapped_llm_explicit_callable(xy_inputs):
#         x_vals = xy_inputs[:, 0]
#         y_vals = xy_inputs[:, 1]
#         z_vals = np.array([llm_z_explicit(x, y) for x, y in zip(x_vals, y_vals)])
#         return np.stack((x_vals, y_vals, z_vals), axis=-1)

#     ranges_explicit = {'x': [-2, 2], 'y': [-2, 2]}
#     pc_explicit_gt = generate_points_from_surface_callable(wrapped_gt_explicit_callable, 'explicit', param_ranges=ranges_explicit, num_points=2500)
#     print(f"Generated Explicit GT PC shape: {pc_explicit_gt.shape if pc_explicit_gt is not None else 'None'}")

#     pc_explicit_llm = generate_points_from_surface_callable(wrapped_llm_explicit_callable, 'explicit', param_ranges=ranges_explicit, num_points=2500)
#     print(f"Generated Explicit LLM PC shape: {pc_explicit_llm.shape if pc_explicit_llm is not None else 'None'}")

#     if pc_explicit_gt is not None and pc_explicit_llm is not None:
#         cd_explicit = compute_chamfer_distance(pc_explicit_gt, pc_explicit_llm)
#         hd_explicit = compute_hausdorff_distance(pc_explicit_gt, pc_explicit_llm)
#         print(f"Chamfer Distance (Explicit): {cd_explicit:.4f}")
#         print(f"Hausdorff Distance (Explicit): {hd_explicit:.4f}")

#     # Example 2: Parametric Surface (Sphere)
#     print("\n--- Testing Parametric Surface (Sphere) ---")
    
#     # Define simple parametric callables for testing
#     def gt_x_param(u, v): return 1.0 * math.cos(u) * math.sin(v)
#     def gt_y_param(u, v): return 1.0 * math.sin(u) * math.sin(v)
#     def gt_z_param(u, v): return 1.0 * math.cos(v)

#     def llm_x_param(u, v): return 0.95 * math.cos(u) * math.sin(v)
#     def llm_y_param(u, v): return 1.05 * math.sin(u) * math.sin(v)
#     def llm_z_param(u, v): return 1.0 * math.cos(v)

#     # Wrapped callables
#     def wrapped_gt_param_callable(uv_inputs):
#         u_vals = uv_inputs[:, 0]
#         v_vals = uv_inputs[:, 1]
#         x_vals = np.array([gt_x_param(u, v) for u, v in zip(u_vals, v_vals)])
#         y_vals = np.array([gt_y_param(u, v) for u, v in zip(u_vals, v_vals)])
#         z_vals = np.array([gt_z_param(u, v) for u, v in zip(u_vals, v_vals)])
#         return np.stack((x_vals, y_vals, z_vals), axis=-1)

#     def wrapped_llm_param_callable(uv_inputs):
#         u_vals = uv_inputs[:, 0]
#         v_vals = uv_inputs[:, 1]
#         x_vals = np.array([llm_x_param(u, v) for u, v in zip(u_vals, v_vals)])
#         y_vals = np.array([llm_y_param(u, v) for u, v in zip(u_vals, v_vals)])
#         z_vals = np.array([llm_z_param(u, v) for u, v in zip(u_vals, v_vals)])
#         return np.stack((x_vals, y_vals, z_vals), axis=-1)

#     ranges_parametric = {'u': [0, 2*np.pi], 'v': [0, np.pi]}
#     pc_parametric_gt = generate_points_from_surface_callable(wrapped_gt_param_callable, 'parametric', param_ranges=ranges_parametric, num_points=2500)
#     print(f"Generated Parametric GT PC shape: {pc_parametric_gt.shape if pc_parametric_gt is not None else 'None'}")

#     pc_parametric_llm = generate_points_from_surface_callable(wrapped_llm_param_callable, 'parametric', param_ranges=ranges_parametric, num_points=2500)
#     print(f"Generated Parametric LLM PC shape: {pc_parametric_llm.shape if pc_parametric_llm is not None else 'None'}")

#     if pc_parametric_gt is not None and pc_parametric_llm is not None:
#         cd_parametric = compute_chamfer_distance(pc_parametric_gt, pc_parametric_llm)
#         hd_parametric = compute_hausdorff_distance(pc_parametric_gt, pc_parametric_llm)
#         print(f"Chamfer Distance (Parametric): {cd_parametric:.4f}")
#         print(f"Hausdorff Distance (Parametric): {hd_parametric:.4f}")

#     # Example 3: Implicit Surface (f = x^2 + y^2 + z^2 - 1)
#     print("\n--- Testing Implicit Surface (f = x^2 + y^2 + z^2 - 1) ---")
    
#     def gt_f_implicit(x, y, z): return x**2 + y**2 + z**2 - 1
#     def llm_f_implicit(x, y, z): return x**2 + y**2 + z**2 - 1.1 # Slightly off

#     ranges_implicit = {'x': [-1.5, 1.5], 'y': [-1.5, 1.5], 'z': [-1.5, 1.5]}
    
#     pc_implicit_gt_simulated = generate_points_from_surface_callable(gt_f_implicit, 'implicit', param_ranges=ranges_implicit, num_points=2500)
#     print(f"Generated Implicit GT PC (simulated) shape: {pc_implicit_gt_simulated.shape if pc_implicit_gt_simulated is not None else 'None'}")

#     pc_implicit_llm_simulated = generate_points_from_surface_callable(llm_f_implicit, 'implicit', param_ranges=ranges_implicit, num_points=2500)
#     print(f"Generated Implicit LLM PC (simulated) shape: {pc_implicit_llm_simulated.shape if pc_implicit_llm_simulated is not None else 'None'}")
    
#     if pc_implicit_gt_simulated is not None and pc_implicit_llm_simulated is not None:
#         cd_implicit = compute_chamfer_distance(pc_implicit_gt_simulated, pc_implicit_llm_simulated)
#         hd_implicit = compute_hausdorff_distance(pc_implicit_gt_simulated, pc_implicit_llm_simulated)
#         print(f"Chamfer Distance (Implicit): {cd_implicit:.4f}")
#         print(f"Hausdorff Distance (Implicit): {hd_implicit:.4f}")
