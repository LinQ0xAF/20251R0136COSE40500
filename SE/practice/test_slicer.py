from bookutils import next_inputs, print_content
import Debugger, inspect
from Slicer import Dependencies, Slicer
from remove_html_markup import remove_html_markup


def middle(x, y, z):  # type: ignore
    if y < z:
        if x < y:
            return y
        elif x < z:
            return y
    else:
        if x > y:
            return y
        elif x > z:
            return x
    return z


with Slicer(middle) as slicer:
    m = middle(2, 1, 3)
m
print(slicer.dependencies())