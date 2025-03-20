from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import os
import re
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from functools import cached_property
from types import TracebackType
from typing import Any, Generator, List, Optional, Sequence, Type, Union, get_args, get_type_hints

import futureexpert


@dataclass
class DocstringSummary:
    """Summary of a docstring.

    Parameters
    ----------
    name
        Name of documented object (can be e.g. a class, method or function).
    docstring
        Docstring of object called `name`.
    refined_docstring
        Possibly refined version of the original docstring.
    class_name
        Name of corresponding class, if object called `name` is a method.
    """
    name: str
    docstring: Optional[str]
    refined_docstring: Optional[str] = None
    class_name: Optional[str] = None


def create_project_clone(source, target):
    """Create clone of project from source in target.

    Parameters
    ----------
    source
        Source directory of project to be cloned.
    target
        Target directory to clone source to.
    """

    def onerror_handling(function, path, excinfo):
        """Auxiliary function for rmtree, which suppresses errors in case that the path to delete does not exist."""
        if excinfo[0] is not FileNotFoundError:
            raise excinfo

    shutil.rmtree(target, onerror=onerror_handling)
    _ = shutil.copytree(source, target)


def module_paths(package_dir) -> Generator[str, None, None]:
    """Collect all module paths of python package.

    Parameters
    ----------
    package_dir
        Root directory of python package.

    Returns
    -------
        Generator providing paths of all python package modules.
    """
    for subdir, dirs, files in os.walk(os.path.join(package_dir, 'futureexpert')):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".py") and not file.startswith('__'):
                yield filepath


class DocstringRefiner:
    """Class bundling functionality to refine docstrings in python code."""

    def __init__(self, module_path: str):
        """Initializer.

        Parameters
        ----------
        module_path
            Path to module to be refined.
        """
        self.module_path = module_path
        self._load_module_tree()
        self.docstrings: List[DocstringSummary] = []

    def _load_module_tree(self):
        """Load tree representation of module."""
        with open(self.module_path, 'r') as f:
            file_contents = f.read()
        self.module_tree = ast.parse(file_contents)

    def load_docstrings(self):
        """Load docstrings of module."""
        for node in self.module_tree.body:
            if isinstance(node, ast.FunctionDef):
                self.docstrings.append(DocstringSummary(name=node.name, docstring=ast.get_docstring(node, clean=False)))
            elif isinstance(node, ast.ClassDef):
                self.docstrings.append(DocstringSummary(name=node.name, docstring=ast.get_docstring(node, clean=False)))
                for class_node in node.body:
                    if isinstance(class_node, ast.FunctionDef):
                        self.docstrings.append(DocstringSummary(name=class_node.name,
                                                                docstring=ast.get_docstring(class_node, clean=False),
                                                                class_name=node.name))

    def get_type_hint_expression(self, type_hint: dict[str, Any]) -> str:
        """Takes type hint and creates a well formatted string."""
        # types with __origin__ values need special treatment. They are more complex types,
        # usually Union, Sequence, or Optional. With very few exceptions (Hashable), they
        # contain more attributes, such as Sequence[int] or Optional[float].
        if not hasattr(type_hint, '__origin__'):
            return f"{type_hint.__module__}.{type_hint.__name__}"
        args = get_args(type_hint)
        if len(args) == 0:
            return f"{type_hint.__module__}.{type_hint.__origin__.__name__}"
        # Optional fields are treated as Unions[RelevantType, None]. Make sure we report "Optional"
        # instead of "Union" and only the relevant type, not the "None" of that Union.
        is_optional = (type_hint.__origin__ is Union and
                       len(args) == 2 and
                       args[1] is type(None))
        origin_name = 'Optional' if is_optional else type_hint.__origin__.__name__
        relevant_args = [args[0]] if is_optional else [arg for arg in args if arg is not type(None)]
        args_str = ', '.join(
            self.get_type_hint_expression(arg)
            for arg in relevant_args
        )
        return f"{type_hint.__module__}.{origin_name}[{args_str}]"

    def refine_docstrings(self):
        """Refine docstrings of module."""

        # Reconstruct module path from file path, to import module and extract type hints.
        module_dot_path = re.sub(r'.*(futureexpert/.*).py', r'\g<1>', self.module_path).replace('/', '.')

        # Import module itself.
        module = importlib.import_module(module_dot_path, package='futureexpert')

        for doc in self.docstrings:

            if doc.docstring is None:
                continue
            else:
                doc.refined_docstring = doc.docstring

            # Get type hints.
            try:
                if doc.class_name is None:
                    type_hints = get_type_hints(obj=getattr(module, doc.name))
                else:
                    obj = getattr(getattr(module, doc.class_name), doc.name)
                    if isinstance(obj, property) or isinstance(obj, cached_property):
                        continue
                    else:
                        type_hints = get_type_hints(obj=obj)
            except Exception as exc:
                message = f'Getting type hints for definition {doc.name} in module {module_dot_path} failed due to {exc}'
                warnings.warn(message)
                continue
            self.add_type_information(doc, type_hints)

    def add_type_information(self, doc, type_hints):
        """Adds type information for parameters of a type. If a parameter is missing from the docstring, it adds the
        name and type of that parameter without any description."""
        for parameter_name, type_hint in type_hints.items():
            # Try to supplement the parameter description in the docstring with the corresponding type hint.
            try:
                type_hint_expression = self.get_type_hint_expression(type_hint)
            except Exception:
                type_hint_expression = str(type_hint)

            to_replace = parameter_name + '\n'
            replacement = parameter_name + ': ' + type_hint_expression + '\n'
            # If a parameter is not mentioned in the docstring, we append them with name+type, without description.
            if to_replace in doc.refined_docstring:
                doc.refined_docstring = doc.refined_docstring.replace(to_replace, replacement)
            else:
                if 'parameters' not in doc.refined_docstring.lower():
                    doc.refined_docstring += '\n\n    Parameters\n    ----------\n    '
                doc.refined_docstring += (replacement + '\n    ')

    def write(self, path: Optional[str] = None):
        """Write module code with refined docstrings.

        Parameters
        ----------
        path
            Path of module to be updated.
        """

        path = path or self.module_path

        with open(path, 'r') as f:
            file_contents = f.read()

        for doc in self.docstrings:
            if (doc.refined_docstring is not None) and (doc.docstring is not None):
                file_contents = file_contents.replace(doc.docstring, doc.refined_docstring)

        with open(path, 'w') as f:
            f.write(file_contents)


class TmpDirectoryManager(object):
    """Auxiliary context manager handling creation and deletion of temporary processing directories."""

    def __init__(self, initiation_directory: str, delete_when_exiting: bool = True) -> None:
        """Initializer.

        Parameters
        ----------
        initiation_directory
            Directory to initialize the new `tmp` directory inside.
        delete_when_exiting
            Whether to delete temporary directory when exiting the context manager.
        """
        self.initiation_directory = initiation_directory
        self.delete_when_exiting = delete_when_exiting
        self.tmp_directory = os.path.join(initiation_directory, 'tmp')

    def __enter__(self) -> TmpDirectoryManager:
        """Enter method of context manager."""
        if not os.path.isdir(self.tmp_directory):
            os.makedirs(self.tmp_directory)
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        """Exit method of context manager."""
        if self.delete_when_exiting:
            shutil.rmtree(self.tmp_directory)


def main(argv: Optional[Sequence[str]] = None):

    futureexpert_dir = os.path.dirname(os.path.dirname(inspect.getfile(futureexpert)))

    parser = argparse.ArgumentParser()
    parser.add_argument('--application-directory', default=futureexpert_dir)
    parser.add_argument('--export-directory', default='/workspaces/futureexpert/public')
    args, _ = parser.parse_known_args(argv)

    application_directory = args.application_directory
    export_directory = args.export_directory

    with TmpDirectoryManager(initiation_directory=os.path.join(tempfile.gettempdir())) as tmp_manager:

        create_project_clone(source=application_directory, target=tmp_manager.tmp_directory)

        for path in module_paths(package_dir=tmp_manager.tmp_directory):

            refiner = DocstringRefiner(module_path=path)
            refiner.load_docstrings()
            refiner.refine_docstrings()
            refiner.write()

        # Generate API docs with pdoc.
        command = f'pdoc --force --html -o {export_directory} {os.path.join(tmp_manager.tmp_directory, "futureexpert")}'
        os.system(command)


if __name__ == '__main__':
    main()
