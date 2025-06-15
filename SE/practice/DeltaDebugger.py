import sys
from typing import Sequence, Any, Callable, Optional, Type, Tuple
from typing import Dict, Union, Set, List, FrozenSet, cast
from types import FunctionType, FrameType, TracebackType
from StackInspector import StackInspector

PASS = 'PASS'
FAIL = 'FAIL'
UNRESOLVED = 'UNRESOLVED'

class NoCallError(ValueError):
    pass
class NotFailingError(ValueError):
    pass
class NotPassingError(ValueError):
    pass
class FailureNotReproducedError(ValueError):
    pass

class CallCollector(StackInspector):
    """
    Collect an exception-raising function call f().
    Use as `with CallCollector(): f()`
    """

    def __init__(self) -> None:
        """Initialize collector"""
        self.init()

    def init(self) -> None:
        """Reset for new collection."""
        self._function: Optional[Callable] = None
        self._args: Dict[str, Any] = {}
        self._exception: Optional[BaseException] = None
        self.original_trace_function: Optional[Callable] = None

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        """Tracing function. Collect first call, then turn tracing off."""
        if event == 'call':
            name = frame.f_code.co_name
            if name.startswith('__'):
                # Internal function
                return
            if self._function is not None:
                # Already set
                return

            func = self.search_func(name, frame)
            if func:
                self._function = func
            else:
                # Create new function from given code
                self._function = self.create_function(frame)

            self._args = {}  # Create a local copy of args
            for var in frame.f_locals:
                if var in frame.f_code.co_freevars:
                    continue  # Local var, not an argument
                self._args[var] = frame.f_locals[var]

            # Turn tracing off
            sys.settrace(self.original_trace_function)

    def after_collection(self) -> None:
        """Called after collection. To be defined in subclasses."""
        pass

    def args(self) -> Dict[str, Any]:
        """Return the dictionary of collected arguments."""
        return self._args

    def function(self) -> Callable:
        """Return the function called."""
        if self._function is None:
            raise NoCallError("No function call collected")
        return self._function

    def exception(self) -> Optional[BaseException]:
        """Return the exception produced, or `None` if none."""
        return self._exception

    def format_call(self, args: Optional[Dict[str, Any]] = None) -> str:  # type: ignore
        ...

    def format_exception(self, exc: Optional[BaseException] = None) -> str:  # type: ignore
        ...

    def call(self, new_args: Optional[Dict[str, Any]] = None) -> Any:  # type: ignore
        ...
        
class CallCollector(CallCollector):
    def __enter__(self) -> Any:
        """Called at begin of `with` block. Turn tracing on."""
        self.init()
        self.original_trace_function = sys.gettrace()
        sys.settrace(self.traceit)
        return self

    def __exit__(self, exc_tp: Type, exc_value: BaseException,
                 exc_traceback: TracebackType) -> Optional[bool]:
        """Called at end of `with` block. Turn tracing off."""
        sys.settrace(self.original_trace_function)

        if not self._function:
            if exc_tp:
                return False  # re-raise exception
            else:
                raise NoCallError("No call collected")

        if self.is_internal_error(exc_tp, exc_value, exc_traceback):
            return False  # Re-raise exception

        self._exception = exc_value
        self.after_collection()
        return True  # Ignore exception
    
    def call(self, new_args: Optional[Dict[str, Any]] = None) -> Any:
        """
        Call collected function. If `new_args` is given,
        override arguments from its {var: value} entries.
        """

        if new_args is None:
            new_args = {}

        args = {}  # Create local copy
        for var in self.args():
            args[var] = self.args()[var]
        for var in new_args:
            args[var] = new_args[var]

        return self.function()(**args)
    
    def format_call(self, args: Optional[Dict[str, Any]] = None) -> str:
        """Return a string representing a call of the function with given args."""
        if args is None:
            args = self.args()
        return self.function().__name__ + "(" + \
            ", ".join(f"{arg}={repr(args[arg])}" for arg in args) + ")"

    def format_exception(self, exc: Optional[BaseException] = None) -> str:
        """Return a string representing the given exception."""
        if exc is None:
            exc = self.exception()
        s = type(exc).__name__
        if str(exc):
            s += ": " + str(exc)
        return s
    
    
class CallReducer(CallCollector):
    def __init__(self, *, log: Union[bool, int] = False) -> None:
        """Initialize. If `log` is True, enable logging."""
        super().__init__()
        self.log = log
        self.reset()

    def reset(self) -> None:
        """Reset the number of tests."""
        self.tests = 0

    def run(self, args: Dict[str, Any]) -> str:
        """
        Run collected function with `args`. Return
        * PASS if no exception occurred
        * FAIL if the collected exception occurred
        * UNRESOLVED if some other exception occurred.
        Not to be used directly; can be overloaded in subclasses.
        """
        try:
            result = self.call(args)
        except Exception as exc:
            self.last_exception = exc
            if (type(exc) == type(self.exception()) and
                    str(exc) == str(self.exception())):
                return FAIL
            else:
                return UNRESOLVED  # Some other failure

        self.last_result = result
        return PASS
    
class CallReducer(CallReducer):
    def test(self, args: Dict[str, Any]) -> str:
        """Like run(), but also log detail and keep statistics."""
        outcome = self.run(args)
        if outcome == PASS:
            detail = ""
        else:
            detail = f" ({self.format_exception(self.last_exception)})"

        self.tests += 1
        if self.log:
            print(f"Test #{self.tests} {self.format_call(args)}: {outcome}{detail}")

        return outcome

    def reduce_arg(self, var_to_be_reduced: str, args: Dict[str, Any]) -> Sequence:
        """
        Determine and return a minimal value for var_to_be_reduced.
        To be overloaded in subclasses.
        """
        return args[var_to_be_reduced]
    
class CachingCallReducer(CallReducer):
    """Like CallReducer, but cache test outcomes."""

    def init(self) -> None:
        super().init()
        self._cache: Dict[FrozenSet, str] = {}

    def test(self, args: Dict[str, Any]) -> str:
        # Create a hashable index
        try:
            index = frozenset((k, v) for k, v in args.items())
        except TypeError:
            index = frozenset()

        if not index:
            # Non-hashable value – do not use cache
            return super().test(args)

        if index in self._cache:
            return self._cache[index]

        outcome = super().test(args)
        self._cache[index] = outcome

        return outcome

def to_set(inp: Sequence) -> Set:
    """Convert inp into a set of indices"""
    return set(range(len(inp)))

def empty(inp: Any) -> Any:
    """Return an "empty" element of the same type as inp"""
    return type(inp)()

def add_to(collection: Any, elem: Any) -> Any:
    """Add element to collection; return new collection."""
    if isinstance(collection, str):
        return collection + elem  # Strings

    try:  # Lists and other collections
        return collection + type(collection)([elem])
    except TypeError:
        pass

    try:  # Sets
        return collection | type(collection)([elem])
    except TypeError:
        pass

    raise ValueError("Cannot add element to collection")

def from_set(the_set: Any, inp: Sequence) -> Any:
    """Convert a set of indices into `inp` back into a collection."""
    ret = empty(inp)
    for i, c in enumerate(inp):
        if i in the_set:
            ret = add_to(ret, c)

    return ret

def split(elems: Any, n: int) -> List:
    assert 1 <= n <= len(elems)

    k, m = divmod(len(elems), n)
    try:
        subsets = list(elems[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
                       for i in range(n))
    except TypeError:
        # Convert to list and back
        subsets = list(type(elems)(
                    list(elems)[i * k + min(i, m):(i + 1) * k + min(i + 1, m)])
                       for i in range(n))

    assert len(subsets) == n
    assert sum(len(subset) for subset in subsets) == len(elems)
    assert all(len(subset) > 0 for subset in subsets)

    return subsets

def is_reducible(value: Any) -> bool:
    # Return True if `value` supports len() and indexing.
    try:
        _ = len(value)
    except TypeError:
        return False

    try:
        _ = value[0]
    except TypeError:
        return False
    except IndexError:
        return False

    return True

class DeltaDebugger(CachingCallReducer):
    def dd(self, var_to_be_reduced: str, fail_args: Dict[str, Any], 
           *, mode: str = '-') -> Tuple[Sequence, Sequence, Sequence]:
        """General Delta Debugging.
        `var_to_be_reduced` - the name of the variable to reduce.
        `fail_args` - a dict of (failure-inducing) function arguments, 
            with `fail_args[var_to_be_reduced]` - the input to apply dd on.
        `mode`- how the algorithm should operate:
            '-' (default): minimize input (`ddmin`),
            '+': maximizing input (`ddmax`),
            '+-': minimizing pass/fail difference (`dd`)
        Returns a triple (`pass`, `fail`, `diff`) with
        * maximized passing input (`pass`), 
        * minimized failing input (`fail`), and
        * their difference `diff`
          (elems that are in `fail`, but not in `pass`).
        """
        def test(c: Set) -> str:
            # Set up args
            test_args = {}
            for var in fail_args:
                test_args[var] = fail_args[var]
            test_args[var_to_be_reduced] = from_set(c, fail_inp)
            return self.test(test_args)

        def ret(c_pass: Set, c_fail: Set) -> \
            Tuple[Sequence, Sequence, Sequence]:
            return (from_set(c_pass, fail_inp),
                    from_set(c_fail, fail_inp),
                    from_set(c_fail - c_pass, fail_inp))

        n = 2  # Initial granularity

        fail_inp = fail_args[var_to_be_reduced]

        c_pass = to_set([])
        c_fail = to_set(fail_inp)
        offset = 0

        minimize_fail = '-' in mode
        maximize_pass = '+' in mode

        # Validate inputs
        if test(c_pass) == FAIL:
            if maximize_pass:
                s_pass = repr(from_set(c_pass, fail_inp))
                raise NotPassingError(
                    f"Input {s_pass} expected to pass, but fails")
            else:
                return ret(c_pass, c_pass)

        if test(c_fail) == PASS:
            if minimize_fail:
                s_fail = repr(from_set(c_fail, fail_inp))
                raise NotFailingError(
                    f"Input {s_fail} expected to fail, but passes")
            else:
                return ret(c_fail, c_fail)

        # Main loop
        while True:
            if self.log > 1:
                print("Passing input:", repr(from_set(c_pass, fail_inp)))
                print("Failing input:", repr(from_set(c_fail, fail_inp)))
                print("Granularity:  ", n)

            delta = c_fail - c_pass
            if len(delta) < n:
                return ret(c_pass, c_fail)

            deltas = split(delta, n)

            reduction_found = False
            j = 0

            while j < n:
                i = (j + offset) % n
                next_c_pass = c_pass | deltas[i]
                next_c_fail = c_fail - deltas[i]

                if minimize_fail and n == 2 and test(next_c_pass) == FAIL:
                    if self.log > 1:
                        print("Reduce to subset")
                    c_fail = next_c_pass
                    offset = i  # was offset = 0 in original dd()
                    reduction_found = True
                    break

                elif maximize_pass and n == 2 and test(next_c_fail) == PASS:
                    if self.log > 1:
                        print("Increase to subset")
                    c_pass = next_c_fail
                    offset = i  # was offset = 0 in original dd()
                    reduction_found = True
                    break

                elif minimize_fail and test(next_c_fail) == FAIL:
                    if self.log > 1:
                        print("Reduce to complement")
                    c_fail = next_c_fail
                    n = max(n - 1, 2)
                    offset = i
                    reduction_found = True
                    break

                elif maximize_pass and test(next_c_pass) == PASS:
                    if self.log > 1:
                        print("Increase to complement")
                    c_pass = next_c_pass
                    n = max(n - 1, 2)
                    offset = i
                    reduction_found = True
                    break

                else:
                    j += 1  # choose next subset

            if not reduction_found:
                if self.log > 1:
                    print("No reduction found")

                if n >= len(delta):
                    return ret(c_pass, c_fail)

                if self.log > 1:
                    print("Increase granularity")

                n = min(n * 2, len(delta))
                
class DeltaDebugger(DeltaDebugger):
    def check_reproducibility(self) -> None:
        # Check whether running the function again fails
        assert self._function, \
            "No call collected. Use `with dd: func()` first."
        assert self._args, \
            "No arguments collected. Use `with dd: func(args)` first."

        self.reset()
        outcome = self.test(self.args())
        if outcome == UNRESOLVED:
            raise FailureNotReproducedError(
                "When called again, " +
                self.format_call(self.args()) + 
                " raised " +
                self.format_exception(self.last_exception) +
                " instead of " +
                self.format_exception(self.exception()))

        if outcome == PASS:
            raise NotFailingError("When called again, " +
                                  self.format_call(self.args()) + 
                                  " did not fail")
        assert outcome == FAIL
        
    def process_args(self, strategy: Callable, **strategy_args: Any) -> \
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Reduce all reducible arguments, using `strategy`(var, `strategy_args`).
        Can be overloaded in subclasses.
        """

        pass_args = {}  # Local copy
        fail_args = {}  # Local copy
        diff_args = {}
        for var in self.args():
            fail_args[var] = self.args()[var]
            diff_args[var] = self.args()[var]
            pass_args[var] = self.args()[var]

            if is_reducible(pass_args[var]):
                pass_args[var] = empty(pass_args[var])

        vars_to_be_processed = set(fail_args.keys())

        pass_processed = 0
        fail_processed = 0

        self.check_reproducibility()

        # We take turns in processing variables until all are processed
        while len(vars_to_be_processed) > 0:
            for var in vars_to_be_processed:
                if not is_reducible(fail_args[var]):
                    vars_to_be_processed.remove(var)
                    break

                if self.log:
                    print(f"Processing {var}...")

                maximized_pass_value, minimized_fail_value, diff = \
                    strategy(var, fail_args, **strategy_args)

                if (maximized_pass_value is not None and 
                    len(maximized_pass_value) > len(pass_args[var])):
                    pass_args[var] = maximized_pass_value
                    # FIXME: diff_args may not be correct for multiple args
                    diff_args[var] = diff
                    if self.log:
                        print(f"Maximized {var} to",
                              repr(maximized_pass_value))
                    vars_to_be_processed = set(fail_args.keys())
                    pass_processed += 1

                if (minimized_fail_value is not None and 
                    len(minimized_fail_value) < len(fail_args[var])):
                    fail_args[var] = minimized_fail_value
                    diff_args[var] = diff
                    if self.log:
                        print(f"Minimized {var} to",
                              repr(minimized_fail_value))
                    vars_to_be_processed = set(fail_args.keys())
                    fail_processed += 1

                vars_to_be_processed.remove(var)
                break

        assert pass_processed == 0 or self.test(pass_args) == PASS, \
            f"{self.format_call(pass_args)} does not pass"
        assert fail_processed == 0 or self.test(fail_args) == FAIL, \
            f"{self.format_call(fail_args)} does not fail"

        if self.log and pass_processed > 0:
            print("Maximized passing call to",
                  self.format_call(pass_args))
        if self.log and fail_processed > 0:
            print("Minimized failing call to",
                  self.format_call(fail_args))

        return pass_args, fail_args, diff_args
    
    def after_collection(self) -> None:
        # Some post-collection checks
        if self._function is None:
            raise NoCallError("No function call observed")
        if self.exception() is None:
            raise NotFailingError(
                f"{self.format_call()} did not raise an exception")

        if self.log:
            print(f"Observed {self.format_call()}" +
                  f" raising {self.format_exception(self.exception())}")
            
    def min_args(self) -> Dict[str, Any]:
        """Return 1-minimal arguments."""
        pass_args, fail_args, diff = self.process_args(self.dd, mode='-')
        return fail_args
    
    def max_args(self) -> Dict[str, Any]:
        """Return 1-maximal arguments."""
        pass_args, fail_args, diff = self.process_args(self.dd, mode='+')
        return pass_args
    
    def min_arg_diff(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Return 1-minimal difference between arguments."""
        return self.process_args(self.dd, mode='+-')
    
    def __repr__(self) -> str:
        """Return a string representation of the minimized call."""
        return self.format_call(self.min_args())
    
'''    
"""test"""
import random

def mystery(inp: str) -> None:
    x = inp.find(chr(0o17 + 0o31))
    y = inp.find(chr(0o27 + 0o22))
    if x >= 0 and y >= 0 and x < y:
        raise ValueError("Invalid input")
    else:
        pass

def fuzz() -> str:
    length = random.randrange(10, 70)
    fuzz = ""
    for i in range(length):
        fuzz += chr(random.randrange(32, 127))
    return fuzz
    
while True:
    fuzz_input = fuzz()
    try:
        mystery(fuzz_input)
    except ValueError:
        break

failing_input = fuzz_input


with DeltaDebugger(log=True) as dd:
    mystery(failing_input)
dd
'''