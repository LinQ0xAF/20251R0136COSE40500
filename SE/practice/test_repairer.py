from Repairer import Repairer, ConditionMutator
from StatisticalDebugger import OchiaiDebugger
import ast, inspect, random
from typing import Any, Callable, Optional, Type, Tuple
from typing import Dict, Union, Set, List, cast
from ExpectError import ExpectError
from bookutils import print_content


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


def remove_html_markup_tree() -> ast.AST:
    return ast.parse(inspect.getsource(remove_html_markup))

def remove_html_markup_test(html: str, plain: str) -> None:
    outcome = remove_html_markup(html)
    assert outcome == plain, \
        f"Got {repr(outcome)}, expected {repr(plain)}"
        
def random_string(length: int = 5, start: int = ord(' '), end: int = ord('~')) -> str:
    return "".join(chr(random.randrange(start, end + 1)) for i in range(length))

random_string()

def random_id(length: int = 2) -> str:
    return random_string(start=ord('a'), end=ord('z'))

random_id()

def random_plain() -> str:
    return random_string().replace('<', '').replace('>', '')

def random_string_noquotes() -> str:
    return random_string().replace('"', '').replace("'", '')

def random_html(depth: int = 0) -> Tuple[str, str]:
    prefix = random_plain()
    tag = random_id()

    if depth > 0:
        html, plain = random_html(depth - 1)
    else:
        html = plain = random_plain()

    attr = random_id()
    value = '"' + random_string_noquotes() + '"'
    postfix = random_plain()

    return f'{prefix}<{tag} {attr}={value}>{html}</{tag}>{postfix}', \
        prefix + plain + postfix
        
random_html()

def remove_html_testcase(expected: bool = True) -> Tuple[str, str]:
    while True:
        html, plain = random_html()
        outcome = (remove_html_markup(html) == plain)
        if outcome == expected:
            return html, plain
        
REMOVE_HTML_TESTS = 100
REMOVE_HTML_PASSING_TESTCASES = \
    [remove_html_testcase(True) for i in range(REMOVE_HTML_TESTS)]
REMOVE_HTML_FAILING_TESTCASES = \
    [remove_html_testcase(False) for i in range(REMOVE_HTML_TESTS)]
# End of Excursion
#print(REMOVE_HTML_PASSING_TESTCASES[0])

#html, plain = REMOVE_HTML_PASSING_TESTCASES[0]
#print(remove_html_markup_test(html, plain))

#print(REMOVE_HTML_FAILING_TESTCASES[0])

"""
with ExpectError():
    html, plain = REMOVE_HTML_FAILING_TESTCASES[0]
    remove_html_markup_test(html, plain)
"""

html_debugger = OchiaiDebugger()
for html, plain in (REMOVE_HTML_PASSING_TESTCASES + 
                    REMOVE_HTML_FAILING_TESTCASES):
    with html_debugger:
        remove_html_markup_test(html, plain)

#print(html_debugger)

html_repairer = Repairer(html_debugger, log=True)
#best_tree, fitness = html_repairer.repair(iterations=20)

# docassert
#assert fitness < 1.0

condition_repairer = Repairer(html_debugger, mutator_class=ConditionMutator, log=2)

best_tree, fitness = condition_repairer.repair(iterations=200)
repaired_source = ast.unparse(best_tree)
print_content(repaired_source, '.py')

# docassert
assert fitness >= 1.0