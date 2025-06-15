# Bookutils

from typing import Any, Dict, List, Set, Optional, Union, Tuple, Type

import sys
import os
import html

# Define the contents of this file as a package
__all__ = [
    "PrettyTable", "YouTubeVideo",
    "print_file", "print_content", "HTML",
    "show_ast", "input", "next_inputs",
    "unicode_escape", "terminal_escape", "project"
    "inheritance_conflicts", "extract_class_definition",
    "quiz", "import_notebooks", "rich_output",
    "InteractiveSVG"
]

# Check for rich output
def rich_output() -> bool:
    try:
        get_ipython()  # type: ignore
        rich = True
    except NameError:
        rich = False

    return rich
    
# Project identifier
def project() -> Optional[str]:
    wd = os.getcwd()
    for name in [ 'fuzzingbook', 'debuggingbook' ]:
        if name in wd:
            return name

    return None


# Wrapper for interactive SVG - like SVG, but with links
# From https://stackoverflow.com/questions/77902970/how-can-i-display-an-interactive-svg-image-that-utilizes-javascript-in-a-jupyter

class InteractiveSVG:
    def __init__(self, filename: str) -> None:
        with open(filename, 'r', encoding='utf-8') as svg_fh:
            self.svg_content = svg_fh.read()
            
    def _repr_html_(self) -> str:
        return self.svg_content


# Checking for inheritance conflicts

# Multiple inheritance is a tricky thing.  If you have two classes $A'$ and $A''$ which both inherit from $A$, the same method $m()$ of $A$ may be overloaded in both $A'$ and $A''$.  If one now inherits from _both_ $A'$ and $A''$, and calls $m()$, which of the $m()$ implementations should be called?  Python "resolves" this conflict by simply invoking the one $m()$ method in the class one inherits from first.
# To avoid such conflicts, one can check whether the order in which one inherits makes a difference.  So try this method to compare the attributes with each other; if they refer to different code, you have to resolve the conflict.

from inspect import getattr_static

def inheritance_conflicts(c1: Type[object], c2: Type[object]) -> List[str]:
    """Return attributes defined differently in classes c1 and c2"""
    class c1c2(c1, c2):  # type: ignore
        pass

    class c2c1(c2, c1):  # type: ignore
        pass

    return [attr for attr in dir(c1c2) if getattr_static(
        c1c2, attr) != getattr_static(c2c1, attr)]

# Printing files with syntax highlighting
def print_file(filename: str, **kwargs: Any) -> None:
    content = open(filename, "rb").read().decode('utf-8')
    print_content(content, filename, **kwargs)

def print_content(content: str, filename: Optional[str] = None, lexer: Optional[Any] = None, start_line_number: Optional[int] = None) -> None:
    from pygments import highlight, lexers, formatters
    from pygments.lexers import get_lexer_for_filename, guess_lexer

    if rich_output():
        if lexer is None:
            if filename is None:
                lexer = guess_lexer(content)
            else:
                lexer = get_lexer_for_filename(filename)

        colorful_content = highlight(
            content, lexer,
            formatters.TerminalFormatter())
        content = colorful_content.rstrip()

    if start_line_number is None:
        print(content, end="")
    else:
        content_list = content.split("\n")
        no_of_lines = len(content_list)
        size_of_lines_nums = len(str(start_line_number + no_of_lines))
        for i, line in enumerate(content_list):
            content_list[i] = ('{0:' + str(size_of_lines_nums) + '} ').format(i + start_line_number) + " " + line
        content_with_line_no = '\n'.join(content_list)
        print(content_with_line_no, end="")

def getsourcelines(function: Any) -> Tuple[List[str], int]:
    """A replacement for inspect.getsourcelines(), but with syntax highlighting"""
    import inspect
    
    source_lines, starting_line_number = \
       inspect.getsourcelines(function)
       
    if not rich_output():
        return source_lines, starting_line_number
        
    from pygments import highlight, lexers, formatters
    from pygments.lexers import get_lexer_for_filename
    
    lexer = get_lexer_for_filename('.py')
    colorful_content = highlight(
        "".join(source_lines), lexer,
        formatters.TerminalFormatter())
    content = colorful_content.strip()
    return [line + '\n' for line in content.split('\n')], starting_line_number

from ast import AST

# Showing ASTs
def show_ast(tree: AST) -> Optional[Any]:
    import ast  # Textual alternative111
    print(ast.dump(tree))
    return None

# Escaping unicode characters into ASCII for user-facing strings
def unicode_escape(s: str, error: str = 'backslashreplace') -> str:
    def ascii_chr(byte: int) -> str:
        if 0 <= byte <= 127:
            return chr(byte)
        return r"\x%02x" % byte

    bytes = s.encode('utf-8', error)
    return "".join(map(ascii_chr, bytes))

# Same, but escaping unicode only if output is not a terminal
def terminal_escape(s: str) -> str:
    if rich_output():
        return s
    return unicode_escape(s)


import html
# Interactive inputs. We simulate them by assigning to the global variable INPUTS.

INPUTS: List[str] = []

original_input = input

def input(prompt: str) -> str:
    given_input = None
    try:
        global INPUTS
        given_input = INPUTS[0]
        INPUTS = INPUTS[1:]
    except:
        pass
    
    if given_input:
        if rich_output():
            from IPython.display import display
            display(html(f"<samp>{prompt}<b>{given_input}</b></samp>"))
        else:
            print(f"{prompt} {given_input}")
        return given_input
    
    return original_input(prompt)
    
def next_inputs(list: List[str] = []) -> List[str]:
    global INPUTS
    INPUTS += list
    return INPUTS
