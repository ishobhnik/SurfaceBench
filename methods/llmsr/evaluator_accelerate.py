import ast
import numpy as np 
import warnings


def add_numba_decorator(
        program: str,
        function_to_evolve: str,
) -> str:
    """
    Accelerates code evaluation by adding @numba.jit() decorator to the target function.
z
    Note: Not all NumPy functions are compatible with Numba acceleration.

    Example:
    Input:  def func(a: np.ndarray): return a * 2
    Output: @numba.jit()
            def func(a: np.ndarray): return a * 2
    """
    # parse to syntax tree
    try:
        tree = ast.parse(program)
    except SyntaxError as e:
        warnings.warn(f"Failed to parse program for Numba decoration: {e}. Skipping Numba acceleration.")
        return program

    # check if 'import numba' already exists
    numba_imported = False
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == 'numba' for alias in node.names):
            numba_imported = True
            break
        if isinstance(node, ast.ImportFrom) and node.module == 'numba' and any(alias.name == 'jit' for alias in node.names):
            numba_imported = True
            break

    # add 'import numba' to the top of the program
    if not numba_imported:
        import_node = ast.Import(names=[ast.alias(name='numba', asname=None)])
        tree.body.insert(0, import_node)

    # traverse the tree, and find the function_to_run
    found_target_function = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_to_evolve:
            found_target_function = True
            # the @numba.jit() decorator instance
            decorator = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='numba', ctx=ast.Load()),
                    attr='jit',
                    ctx=ast.Load()
                ),
                args=[],  
                keywords=[ast.keyword(arg='nopython', value=ast.Constant(value=True))]
            )
            # add the decorator to the decorator_list of the node
            node.decorator_list.append(decorator)
            break
    
    if not found_target_function:
        warnings.warn(f"Function '{function_to_evolve}' not found for Numba decoration. Skipping acceleration.")
        return program

    # turn the tree to string and return
    try:
        modified_program = ast.unparse(tree)
        return modified_program
    except Exception as e:
        warnings.warn(f"Failed to unparse AST after Numba decoration: {e}. Skipping Numba acceleration.")
        return program


# if __name__ == '__main__':
#     code = '''
#         import numpy as np
#         # import numba # Test without explicit import
        
#         def func1():
#             return 3

#         def func(a: np.ndarray):
#             return a * 2
#     '''
#     res = add_numba_decorator(code, 'func')
#     print(res)

#     code2 = '''
#         def some_other_func():
#             return 10
#     '''
#     res2 = add_numba_decorator(code2, 'non_existent_func')
#     print(res2)