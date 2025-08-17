from __future__ import annotations
import sys
import os
import time
import shutil
import subprocess
import traceback
import textwrap
import re
import json
import random
import inspect
import ast
import requests

from typing import Any, Dict, List, Tuple, Optional, get_type_hints
from enum import Enum

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Stream handler (stdout)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

from swebase import SWEBase, Patch, Role, Response, ToolCall, \
                    BaseMessage, ToolCallMessage, AssistantMessage


def trim_string_by_lines(text: str, max_lines: int | None = None) -> str:
    """
    Trim a multi-line string to a maximum number of lines.

    Parameters:
        text (str): The string to trim.
        max_lines (int | None): The maximum number of lines to keep.
            If None, no trimming is applied. Defaults to None.

    Returns:
        str: The trimmed string, adding a "(truncated, N more lines)" note if cut.
    """
    if max_lines is None:
        return text

    lines = text.splitlines()
    if len(lines) > max_lines:
        trimmed_count = len(lines) - max_lines
        return "\n".join(lines[:max_lines]) + f"\n... (truncated, {trimmed_count} more lines)"
    return "\n".join(lines)


def tool_metadata(func):
    """
    Generate OpenAI-compatible tool metadata from a Python function.

    This version supports structured return types such as tuples,
    mapping them into a JSON Schema `object` for the "returns" field.

    Parameters:
        func (Callable):
            The function for which to generate metadata.
            Should have type hints for parameters and return value.

    Returns:
        dict:
            Metadata suitable for OpenAI tool definitions, including:
                - function name
                - description from docstring
                - parameters with type and description
                - required parameter list
                - structured or simple return type information

    Type Mapping:
        Python type → JSON Schema type:
            str   → "string"
            int   → "integer"
            float → "number"
            bool  → "boolean"
            other → "string" (fallback)
    """
    def python_type_to_json(ptype):
        """Map Python type to JSON schema type string."""
        if ptype in (str,):
            return "string"
        elif ptype in (int,):
            return "integer"
        elif ptype in (float,):
            return "number"
        elif ptype in (bool,):
            return "boolean"
        else:
            return "string"

    sig = inspect.signature(func)
    hints = get_type_hints(func)
    doc = inspect.getdoc(func) or "No description provided."

    # Parameters metadata
    properties = {}
    required = []
    for name, param in sig.parameters.items():
        ptype = hints.get(name, str)
        ptype_name = getattr(ptype, "__name__", str(ptype))

        properties[name] = {
            "type": python_type_to_json(ptype),
            "description": f"({ptype_name}) Argument '{name}'"
        }
        if param.default is inspect.Parameter.empty:
            required.append(name)

    # Return type metadata
    returns_schema = None
    return_type = hints.get("return", None)
    if return_type:
        # Handle tuple return types
        origin = getattr(return_type, "__origin__", None)
        if origin is tuple or origin is Tuple:
            sub_types = return_type.__args__
            return_properties = {}
            for i, sub_t in enumerate(sub_types):
                sub_t_name = getattr(sub_t, "__name__", str(sub_t))
                return_properties[f"item{i+1}"] = {
                    "type": python_type_to_json(sub_t),
                    "description": f"({sub_t_name}) Return value part {i+1}"
                }
            returns_schema = {
                "type": "object",
                "properties": return_properties,
                "required": list(return_properties.keys())
            }
        else:
            rtype_name = getattr(return_type, "__name__", str(return_type))
            returns_schema = {
                "type": python_type_to_json(return_type),
                "description": f"({rtype_name}) Function return value"
            }

    # Final metadata
    tool_info = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }
    if returns_schema:
        tool_info["function"]["returns"] = returns_schema

    return tool_info


# Directories to skip when scanning for Python files
EXCLUDED_DIRS = {"__pycache__", "doc", "docs", ".git"}

def list_python_files(directory: str, excluded_dirs=None):
    """
    List all Python (.py) files in a directory recursively, excluding certain folders.

    Parameters:
        directory (str):
            The path to the root directory where the search should begin.
        excluded_dirs (set[str] | None):
            A set of directory names to exclude from the search. If None,
            uses the default `EXCLUDED_DIRS` set.

    Returns:
        list[str]:
            A list of full file paths to all discovered `.py` files.
    """
    if excluded_dirs is None:
        excluded_dirs = EXCLUDED_DIRS

    py_files = []
    for root, dirs, files in os.walk(directory):
        # Remove excluded directories from search
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files


def attach_parents(tree):
    """
    Attach a 'parent' attribute to every node in the AST for upward traversal.

    Parameters:
        tree (ast.AST):
            The parsed Abstract Syntax Tree (AST) for a Python file.

    Returns:
        None:
            Modifies the AST nodes in place, adding `.parent` attributes.
    """
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node


def get_qualified_name(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """
    Return qualified name including class/function nesting, e.g.,
    ClassName.method or outer.inner.
    """
    parts = [node.name]
    parent = getattr(node, "parent", None)
    while parent:
        if isinstance(parent, ast.ClassDef):
            parts.insert(0, parent.name)
        elif isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.insert(0, parent.name)
        parent = getattr(parent, "parent", None)
    return ".".join(parts)


def list_functions_in_file(
    filepath: str,
    include_body: bool = False,
    max_lines: int = 1000
) -> List[Dict[str, object]]:
    """
    Parse a Python file and list all functions and methods in a consistent schema.

    Parameters:
        filepath (str): Path to the Python file to analyze.
        include_body (bool, optional): If True, include the function's source code.
                                       Defaults to False.
        max_lines (int, optional): Max lines of function body to include if include_body=True.
                                   Defaults to 1000.

    Returns:
        List[Dict[str, object]]:
            Each dict contains:
                - "type": "function" | "async_function"
                - "name": str, qualified function/method name
                - "start_line": int, start line of definition
                - "end_line": int, end line of definition
                - "body": str | None
    """
    results: List[Dict[str, object]] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        lines = source.splitlines()
        tree = ast.parse(source, filename=filepath)
        attach_parents(tree)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                    func_body = None
                    if include_body:
                        func_code = "\n".join(
                            lines[node.lineno - 1: node.end_lineno]
                        )
                        func_body = trim_string_by_lines(func_code, max_lines)

                    entry = {
                        "type": "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                        "name": get_qualified_name(node),
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "body": func_body,
                    }
                    results.append(entry)

    except (SyntaxError, UnicodeDecodeError) as e:
        results.append({
            "type": "error",
            "name": f"<Error parsing file: {e}>",
            "start_line": -1,
            "end_line": -1,
            "body": None
        })

    # Sort results by line number for consistency
    results.sort(key=lambda x: x["start_line"])
    return results


def list_files_and_functions(
    directory: str,
    include_body: bool = False,
    max_lines: int = 100
) -> Dict[str, List[Dict[str, object]]]:
    """
    List all Python files in a directory and the functions/methods they define.

    Parameters:
        directory (str):
            The path to the root directory to search.
        include_body (bool, optional):
            If True, include function source code in results. Defaults to False.
        max_lines (int, optional):
            Maximum number of lines to include in function bodies. Defaults to 100.

    Returns:
        dict[str, list[dict]]:
            A dictionary mapping each `.py` file path to a list of dictionaries:
            - "type": "function" | "async_function"
            - "name": str, the function or method name
            - "start_line": int, the line number where it starts
            - "end_line": int, the line number where it ends
            - "body": str | None, trimmed function code if included
    """
    result: Dict[str, List[Dict[str, object]]] = {}
    for file_path in list_python_files(directory):
        result[file_path] = list_functions_in_file(
            file_path,
            include_body=include_body,
            max_lines=max_lines
        )
    return result


def grep_with_limit_bash(
    pattern: str,
    search_path: str = ".",
    ignore_case: bool = False,
    max_lines: Optional[int] = None,
    exclude_dirs: Optional[List[str]] = None,
    only_python_files: bool = True
) -> Tuple[str, int]:
    """
    Run `grep` recursively via bash on the specified path, optionally excluding certain directories,
    searching only in Python files (by default), and return up to `max_lines` of output.

    Parameters:
        pattern (str):
            The search term or regular expression for grep.
        search_path (str):
            The directory or file to search in. Defaults to current directory.
        ignore_case (bool):
            If True, run grep with case-insensitive search (-i).
        max_lines (int, optional):
            Maximum number of lines to return from grep output. Defaults to None (no limit).
        exclude_dirs (List[str], optional):
            List of directory names to exclude from the search.
            Defaults to [".git", "__pycache__", "docs"].
        only_python_files (bool, optional):
            If True, restrict search to `.py` files only. Defaults to True.

    Returns:
        Tuple[str, int]:
            A tuple containing:
            - Trimmed grep output or stderr message (UTF-8 decoded).
            - Exit status code from grep.
    """
    if exclude_dirs is None:
        exclude_dirs = EXCLUDED_DIRS

    grep_cmd = ["grep", "-rn", "--binary-files=without-match"]
    if ignore_case:
        grep_cmd.append("-i")
    for d in exclude_dirs:
        grep_cmd.append(f"--exclude-dir={d}")
    if only_python_files:
        grep_cmd.append("--include=*.py")
    grep_cmd.append(pattern)
    grep_cmd.append(search_path)

    try:
        result = subprocess.run(
            grep_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        if result.returncode == 0:
            output = result.stdout.decode("utf-8", errors="replace")
        else:
            output = result.stderr.decode("utf-8", errors="replace")
        return trim_string_by_lines(output, max_lines), result.returncode
    except FileNotFoundError:
        raise EnvironmentError("grep command not found on this system.")


def extract_code_by_line_range(
    file_path: str,
    line_range: Tuple[Optional[int], Optional[int]],
    include_body: bool = True,
    max_lines: int = 5000
) -> List[Dict[str, object]]:
    """
    Extract all function and method definitions in a Python file that overlap with
    the specified line range. Also includes global (non-function) code lines.

    A function or method is included if any part of its body intersects the given range.

    Parameters:
        file_path (str): Path to the Python file to parse.
        line_range (Tuple[Optional[int], Optional[int]]):
            A tuple (start_line, end_line), both inclusive.
            - If start_line is None, it means start of file.
            - If end_line is None, it means end of file.
        include_body (bool, optional): If True, include function/method source code.
                                       Defaults to True.
        max_lines (int, optional): Maximum number of lines to include in each body.
                                   Defaults to 5000. Only used if include_body is True.

    Returns:
        List[Dict[str, object]]:
            Each dict contains:
                - "type": "function" | "async_function" | "global"
                - "name": str, the function/method qualified name or global line
                - "start_line": int, start line number
                - "end_line": int, end line number
                - "body": str | None (always None for globals)
    """
    start_line, end_line = line_range

    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    lines = source.splitlines()

    total_lines = len(lines)
    if start_line is None:
        start_line = 1
    if end_line is None:
        end_line = total_lines

    tree = ast.parse(source)
    attach_parents(tree)
    results: List[Dict[str, object]] = []

    # Functions and methods
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                if not (node.end_lineno < start_line or node.lineno > end_line):
                    func_body = None
                    if include_body:
                        func_code = "\n".join(
                            lines[node.lineno - 1: node.end_lineno]
                        )
                        func_body = trim_string_by_lines(func_code, max_lines)

                    entry = {
                        "type": "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                        "name": get_qualified_name(node),
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "body": func_body,
                    }
                    results.append(entry)

    # Global-level matches (outside functions)
    for i in range(start_line, end_line + 1):
        if i <= len(lines):
            # skip if line is inside any known function/method
            inside_func = any(e["start_line"] <= i <= e["end_line"] for e in results)
            if not inside_func:
                entry = {
                    "type": "global",
                    "name": lines[i - 1].strip(),
                    "start_line": i,
                    "end_line": i,
                    "body": None
                }
                results.append(entry)

    results.sort(key=lambda x: x["start_line"])
    return results


def extract_code_with_search_term(
    filename: str,
    search_term: str,
    ignore_case: bool = False,
    include_body: bool = True,
    max_lines: int = 1000
) -> List[Dict[str, object]]:
    """
    Extract functions, methods, and global-level code from a file that contain
    a given search term.

    Parameters:
        filename (str): Path to the Python file to analyze.
        search_term (str): Term to look for inside functions, methods, and globals.
        ignore_case (bool, optional): If True, search is case-insensitive. Defaults to False.
        include_body (bool, optional): If True, include source code in results. Defaults to True.
        max_lines (int, optional): Max lines of function/method body to include.
                                   Defaults to 1000. Not applied to globals.

    Returns:
        List[Dict[str, object]]:
            Each dict contains:
                - "type": "function" | "async_function" | "global"
                - "name": str, qualified function/method name or first global line
                - "start_line": int, start line number
                - "end_line": int, end line number
                - "body": str | None (globals only if include_body=True, full line included)
    """
    with open(filename, "r", encoding="utf-8") as f:
        source = f.read()
    lines = source.splitlines()

    if ignore_case:
        search_cmp = search_term.lower()
        match_in_line = lambda l: search_cmp in l.lower()
    else:
        match_in_line = lambda l: search_term in l

    tree = ast.parse(source)
    attach_parents(tree)
    results: List[Dict[str, object]] = []

    # Functions and methods
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                func_code = "\n".join(lines[node.lineno - 1: node.end_lineno])
                if match_in_line(func_code):
                    func_body = None
                    if include_body:
                        func_body = trim_string_by_lines(func_code, max_lines)

                    entry = {
                        "type": "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                        "name": get_qualified_name(node),
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "body": func_body,
                    }
                    results.append(entry)

    # Global-level matches
    for i, line in enumerate(lines, 1):
        if match_in_line(line):
            entry = {
                "type": "global",
                "name": line.strip(),
                "start_line": i,
                "end_line": i,
                "body": line if include_body else None
            }
            results.append(entry)

    results.sort(key=lambda x: x["start_line"])
    return results


def search_term_in_directory(
    directory: str,
    search_term: str,
    ignore_case: bool = False,
    include_body: bool = True,
    max_lines: int = 1000
) -> Dict[str, List[Dict[str, object]]]:
    """
    Search for a term in Python files inside a directory using extract_code_with_search_term.

    Parameters:
        directory (str): Path to directory to scan.
        search_term (str): Term to search inside Python code.
        ignore_case (bool, optional): Case-insensitive search. Defaults to False.
        include_body (bool, optional): Include code bodies in results. Defaults to True.
        max_lines (int, optional): Max lines of body to include (functions only). Defaults to 1000.

    Returns:
        dict[str, list[dict]]: Mapping from file path -> list of matches.
    """
    results = {}
    for file_path in list_python_files(directory):
        matches = extract_code_with_search_term(
            file_path,
            search_term,
            ignore_case=ignore_case,
            include_body=include_body,
            max_lines=max_lines
        )
        if matches:
            results[file_path] = matches
    return results


def finalize_test_collection(
    file_to_tests: Dict[str, List[str]], 
    max_lines: Optional[int] = None
) -> Dict[str, Dict[str, str]]:
    """
    Finalize the collection of test functions by retrieving their full source bodies.

    Parameters:
        file_to_tests (dict[str, list[str]]):
            A dictionary mapping Python file paths to lists of test function names
            contained in those files.

            Example:
                {
                    "tests/test_example.py": ["test_add", "test_subtract"],
                    "tests/test_api.py": ["test_get_user"]
                }

        max_lines (int, optional):
            Maximum number of lines for each function body. 
            If None, no trimming is applied. Defaults to None.

    Returns:
        dict[str, dict[str, str]]:
            A dictionary where:
            - Key = file path
            - Value = dictionary mapping test function names to their source code bodies.
    """
    results: Dict[str, Dict[str, str]] = {}

    for file_path, test_names in file_to_tests.items():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=file_path)
            lines = source.splitlines()

            file_results: Dict[str, str] = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in test_names and hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                        func_body = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                        if max_lines is not None:
                            func_body = trim_string_by_lines(func_body, max_lines)
                        file_results[node.name] = func_body

            if file_results:
                results[file_path] = file_results

        except (FileNotFoundError, SyntaxError, UnicodeDecodeError) as e:
            results[file_path] = {f"<Error reading file: {e}>": ""}

    return results


TEST_COLLECTION_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""
# Hey there! You're a Coding Assistant 🚀. I have uploaded all files of a python repository. Your current working directory is at the root of that repo. You will be provided with a problem statement and your task is to find specific test functions specifically designed to test the issue mentioned in the problem statement leveraging the tools provided to you.

You must follow these steps to find the test functions:
1. Read the problem statement carefully and understand the issue.
2. Use grep_with_limit_bash tool to search through the codebase with multiple keywords(individually, combining them logically, etc).
3. Check functions that is related to the issue mentioned in the problem statement and if you find any function that is specifically designed to test the issue follow these steps:
    - Confirm that the function is specifically designed to test the scenario mentioned in the problem statement. Even if the function is directly related to the issue, it might not be specifically designed to test the scenario and you MUST NOT add it to the list of test functions.
    - Once you're confirmed, add that new function to the list of test functions.
    - Check if the current list (including the new function) of test functions are enough already to confirm scenario in the issue is covered.
    - If yes, then you can finish the task and return the list of test functions along with the file path even if less than 3 test functions are in the list.
    - Otherwise, you should continue to search through the codebase until list of test functions covers the issue mentioned in the problem statement or you've found more than 3 test functions already.
4. Check each test function again and filter out the ones that are not specifically designed to test the issue mentioned in the problem statement, but for some general purposes.
5. If there is no test functions after filtered in the list, then you MUST select the most relevant one and add it to the list of test functions to make sure test function list is not empty.
7. Return the list of test functions along with the file path using finalize_test_collection tool.

When you respond, please describe your reasoning process in detail about tool usages and next plan.
""")

TEST_COLLECTION_USER_PROMPT_TEMPLATE = textwrap.dedent("""
# Now let's start. Here is the problem statement:
{problem_statement}
""")


# LLM_MODELS = ["z-ai/glm-4.5", "anthropic/claude-sonnet-4", "deepseek/deepseek-chat-v3-0324", "openai/gpt-5"]
LLM_MODEL = "z-ai/glm-4.5"

STEPS_MAX_TEST_COLLECTION = 100
TIMEOUT_TEST_COLLECTION = 1000


class ResponseErrorType(Enum):
    EMPTY_RESPONSE = 1
    RATE_LIMIT_EXCEEDED = 2
    INVALID_RESPONSE_FORMAT = 3
    TIMEOUT = 4
    UNKNOWN = 5


def summarize_history(history: list[BaseMessage]) -> list[BaseMessage]:
    return history


class SWE(SWEBase):
    def __init__(self):
        super().__init__()

    def call_llm(self, messages: list[BaseMessage], tools: list[dict]) -> Response:
        try:
            response = self.llm.call(
                messages=messages,
                tools=tools,
                temperature=0.0,
                max_tokens=65536,
                model=LLM_MODEL
            )
        except Exception as e:
            error_msg = str(e)
            logger.debug(f"LLM request raises Exception {error_msg}")
            return None

    def run_test_collection(
        self,
        issue_description: str
    ) -> Dict[str, Dict[str, str]]:
        tool_func_list = [
            grep_with_limit_bash,
            extract_code_by_line_range,
            extract_code_with_search_term,
            search_term_in_directory,
            finalize_test_collection
        ]
        tool_func_dict = {k.__name__: k for k in tool_func_list}
        tools = [tool_metadata(k) for k in tool_func_list]

        system_prompt = TEST_COLLECTION_SYSTEM_PROMPT_TEMPLATE
        user_prompt = TEST_COLLECTION_USER_PROMPT_TEMPLATE.format(problem_statement=issue_description)

        history: list[BaseMessage] = []
        step = 0
        start_t = time.perf_counter()
        while step < STEPS_MAX_TEST_COLLECTION:
            step += 1
            now_t = time.perf_counter()
            if (now_t - start_t > TIMEOUT_TEST_COLLECTION):
                logger.debug(f"{step}. timeout occurred during test collection.")
                break

            messages: list[BaseMessage] = [
                BaseMessage(role=Role.SYSTEM, content=system_prompt),
                BaseMessage(role=Role.USER, content=user_prompt)
            ]
            messages.extend(history)

            response = self.call_llm(messages, tools)
            if response.tool_calls is None:
                logger.debug(f"{step}. tool call is None.")
                break

            history.append(AssistantMessage(
                content=response.result,
                tool_calls=response.tool_calls
            ))

            for tc in response.tool_calls:
                if not tc.name in tool_func_dict.keys():
                    continue
                result = tool_func_dict[tc.name](**tc.args)
                if tc.name == "finalize_test_collection":
                    logger.debug(f"{step}. tool call is the finalization.")
                    logger.debug(f"{result}")
                    return result
                history.append(ToolCallMessage(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=json.dumps(result)
                ))

            history = summarize_history(history)

        return None

    def run_produce_patches(
        self,
        issue_description: str,
        file_test_bodies: Dict[str, Dict[str, str]]
    ) -> Patch:
        pass

    def __call__(self, repo_location: str, issue_description: str) -> Patch:
        os.chdir(repo_location)
        file_test_bodies = self.run_test_collection(issue_description)
        return None
