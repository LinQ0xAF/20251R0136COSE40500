from Tracer import Tracer
from Debugger import Debugger
from bookutils import input, next_inputs
import inspect

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


# ignore
next_inputs(["help", "quit"])

with Debugger():
    remove_html_markup('abc')

assert not next_inputs()
