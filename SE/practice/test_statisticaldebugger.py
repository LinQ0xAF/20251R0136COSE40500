from StatisticalDebugger import Collector, CoverageCollector, StatisticalDebugger
import inspect
from bookutils import getsourcelines
from typing import Any, Callable, Optional, Type, Tuple
from typing import Dict, Set, List, TypeVar, Union
    
def remove_html_markup(s):  # type: ignore
    tag = False
    quote = False
    out = ""

    for c in s:
        if c == '<' and not quote:
            tag = True
        elif c == '>' and not quote:
            tag = False
        elif c == '"' or c == "'" and tag:
            quote = not quote
        elif not tag:
            out = out + c

    return out

Coverage = Set[Tuple[Callable, int]]
def code_with_coverage(function: Callable, coverage: Coverage) -> None:
    source_lines, starting_line_number = \
       getsourcelines(function)

    line_number = starting_line_number
    for line in source_lines:
        marker = '*' if (function, line_number) in coverage else ' '
        print(f"{line_number:4} {marker} {line}", end='')
        line_number += 1
        
s = StatisticalDebugger()
with s.collect('PASS'):
    remove_html_markup("abc")
with s.collect('PASS'):
    remove_html_markup('<b>abc</b>')
with s.collect('FAIL'):
    remove_html_markup('"abc"')