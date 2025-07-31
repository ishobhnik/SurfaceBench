# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Tools for manipulation of Python code samples. Two main classes:
- Function: Represents a function with its name, args, body, and optional return type and docstring.
- Program: Contains a code preface (imports, global variables, classes) and a list of Functions.

Note: 'call' refers to the function name throughout this module.
"""

from __future__ import annotations
from typing import Iterator, MutableSet, Sequence, Any, Optional
from collections.abc import Iterator, MutableSet, Sequence
import dataclasses
import io
import ast
import tokenize
import warnings


@dataclasses.dataclass
class Function:
    """ A parsed Python function. """

    name: str
    args: str
    body: str
    return_type: str | None = None
    docstring: str | None = None
    score: float | None = None
    global_sample_nums: int | None = None  
    sample_time: float | None = None  
    evaluate_time: float | None = None  
    scores_per_test: Optional[Any] = None # Added to store detailed scores per test (Chamfer, Hausdorff)

    def __str__(self) -> str:
        return_type_str = f' -> {self.return_type}' if self.return_type else ''

        function = f'def {self.name}({self.args}){return_type_str}:\n'
        if self.docstring:
            new_line = '\n' if self.body else ''
            function += f'    """{self.docstring}"""{new_line}'

        function += self.body.strip()
        
        return function + '\n\n'

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'body':
            value = str(value).strip('\n')

        if name == 'docstring' and value is not None:
            value = str(value).strip() 
            value = value.replace('"""', '')
        super().__setattr__(name, value)


@dataclasses.dataclass(frozen=True)
class Program:
    """ A parsed Python program."""

    preface: str
    functions: list[Function]


    def __str__(self) -> str:
        program = f'{self.preface}\n' if self.preface else ''
        program += '\n'.join([str(f) for f in self.functions])
        
        return program


    def find_function_index(self, function_name: str) -> int:
        """ Return the index of input function name. """
        function_names = [f.name for f in self.functions]
        count = function_names.count(function_name)
        if count == 0:
            raise ValueError(
                f'function {function_name} does not exist in program:\n{str(self)}'
            )
        if count > 1:
            raise ValueError(
                f'function {function_name} exists more than once in program:\n'
                f'{str(self)}'
            )
        index = function_names.index(function_name)
        
        return index


    def get_function(self, function_name: str) -> Function:
        index = self.find_function_index(function_name)
        
        return self.functions[index]


class ProgramVisitor(ast.NodeVisitor):
    """ Parse code to collect all required information and produce a `Program`. """

    def __init__(self, sourcecode: str):
        self._codelines: list[str] = sourcecode.splitlines()
        self._preface: str = ''
        self._functions: list[Function] = []
        self._current_function: str | None = None


    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """ Collect all information about the function being parsed. """
        if node.col_offset == 0:  
            self._current_function = node.name

            if not self._functions:
                has_decorators = bool(node.decorator_list)
                if has_decorators:
                    decorator_start_line = min(decorator.lineno for decorator in node.decorator_list)
                    self._preface = '\n'.join(self._codelines[:decorator_start_line - 1])
                else:
                    self._preface = '\n'.join(self._codelines[:node.lineno - 1])

            function_end_line = node.end_lineno
            body_start_line = node.body[0].lineno - 1
            
            # Extract the docstring.
            docstring = None
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant)):
                docstring_node = node.body[0].value
                docstring = docstring_node.s if isinstance(docstring_node, ast.Str) else docstring_node.value
                docstring = str(docstring).strip()
                if len(node.body) > 1:
                    body_start_line = node.body[1].lineno - 1
                else:
                    body_start_line = function_end_line

            self._functions.append(Function(
                name=node.name,
                args=ast.unparse(node.args), 
                return_type=ast.unparse(node.returns) if node.returns else None,
                docstring=docstring,
                body='\n'.join(self._codelines[body_start_line:function_end_line]),
            ))
            
        self.generic_visit(node)

    def return_program(self) -> Program:
        return Program(preface=self._preface, functions=self._functions)


def text_to_program(text: str) -> Program:
    """ Return Program object by parsing input text using Python AST. """
    try:
        tree = ast.parse(text)
        visitor = ProgramVisitor(text)
        visitor.visit(tree)
        return visitor.return_program()
    
    except Exception as e:
        warnings.warn(f'Failed parsing program: {text[:100]}... Error: {e}')
        raise e


def text_to_function(text: str) -> Function:
    """ Return Function object by parsing input text using Python AST. """
    program = text_to_program(text)
    
    if len(program.functions) != 1:
        raise ValueError(f'Only one function expected, got {len(program.functions)}'
                         f':\n{program.functions}')
    
    return program.functions[0]


def _tokenize(code: str) -> Iterator[tokenize.TokenInfo]:
    """Transform `code` into Python tokens."""
    code_bytes = code.encode('utf-8')
    code_io = io.BytesIO(code_bytes)
    return tokenize.tokenize(code_io.readline)


def _untokenize(tokens: Sequence[tokenize.TokenInfo]) -> str:
    """Transform a list of Python tokens into code."""
    code_bytes = tokenize.untokenize(tokens)
    return code_bytes.decode('utf-8')

def _yield_token_and_is_call(code: str) -> Iterator[tuple[tokenize.TokenInfo, bool]]:
    """ Yield each token with a bool indicating whether it is a function call. """
    try:
        tokens = _tokenize(code)
        prev_token = None
        is_attribute_access = False
        
        for token in tokens:
            if (prev_token and  
                    prev_token.type == tokenize.NAME and  
                    token.type == tokenize.OP and  
                    token.string == '('):  
                yield prev_token, not is_attribute_access
                is_attribute_access = False
            else:
                if prev_token:
                    is_attribute_access = (
                            prev_token.type == tokenize.OP and prev_token.string == '.'
                    )
                    yield prev_token, False
            prev_token = token
        
        if prev_token:
            yield prev_token, False
    
    except Exception as e:
        warnings.warn(f'Failed parsing code for tokens: {code[:100]}... Error: {e}')
        raise e


def rename_function_calls(code: str, source_name: str, target_name: str) -> str:
    """ Rename function calls from `source_name` to `target_name`. """
    if source_name not in code:
        return code
    modified_tokens = []
    
    for token, is_call in _yield_token_and_is_call(code):
        if is_call and token.string == source_name:
            # Replace the function name token
            modified_token = tokenize.TokenInfo(
                type=token.type,
                string=target_name,
                start=token.start,
                end=token.end,
                line=token.line
            )
            modified_tokens.append(modified_token)
        else:
            modified_tokens.append(token)
    
    return _untokenize(modified_tokens)


def get_functions_called(code: str) -> MutableSet[str]:
    """Return the set of all functions called in `code`. """
    return set(token.string for token, is_call in
               _yield_token_and_is_call(code) if is_call)


def yield_decorated(code: str, module: str, name: str) -> Iterator[str]:
    """Yield names of functions decorated with `@module.name` in `code`. """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                attribute = None
                if isinstance(decorator, ast.Attribute):
                    attribute = decorator
                elif isinstance(decorator, ast.Call):
                    attribute = decorator.func
                if (attribute is not None
                        and isinstance(attribute.value, ast.Name)
                        and attribute.value.id == module
                        and attribute.attr == name):
                    yield node.name