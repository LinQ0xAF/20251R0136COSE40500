import ast, inspect, copy, warnings, random, re, traceback
from bookutils import print_content, show_ast
from ast import NodeVisitor, NodeTransformer
from typing import Any, Callable, Optional, Type, Tuple
from typing import Dict, Union, Set, List, cast
from StackInspector import StackInspector
from StatisticalDebugger import RankingDebugger
from DeltaDebugger import DeltaDebugger


POPULATION_SIZE = 40
WEIGHT_PASSING = 0.99
WEIGHT_FAILING = 0.01

class StatementVisitor(NodeVisitor):
    """Visit all statements within function defs in an AST"""

    def __init__(self) -> None:
        self.statements: List[Tuple[ast.AST, str]] = []
        self.func_name = ""
        self.statements_seen: Set[Tuple[ast.AST, str]] = set()
        super().__init__()

    def add_statements(self, node: ast.AST, attr: str) -> None:
        elems: List[ast.AST] = getattr(node, attr, [])
        if not isinstance(elems, list):
            elems = [elems]  # type: ignore

        for elem in elems:
            stmt = (elem, self.func_name)
            if stmt in self.statements_seen:
                continue

            self.statements.append(stmt)
            self.statements_seen.add(stmt)

    def visit_node(self, node: ast.AST) -> None:
        # Any node other than the ones listed below
        self.add_statements(node, 'body')
        self.add_statements(node, 'orelse')

    def visit_Module(self, node: ast.Module) -> None:
        # Module children are defs, classes and globals - don't add
        super().generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Class children are defs and globals - don't add
        super().generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        self.visit_node(node)
        super().generic_visit(node)

    def visit_FunctionDef(self,
                          node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        if not self.func_name:
            self.func_name = node.name

        self.visit_node(node)
        super().generic_visit(node)
        self.func_name = ""

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return self.visit_FunctionDef(node)
    
def all_statements_and_functions(tree: ast.AST, 
                                 tp: Optional[Type] = None) -> \
                                 List[Tuple[ast.AST, str]]:
    """
    Return a list of pairs (`statement`, `function`) for all statements in `tree`.
    If `tp` is given, return only statements of that class.
    """

    visitor = StatementVisitor()
    visitor.visit(tree)
    statements = visitor.statements
    if tp is not None:
        statements = [s for s in statements if isinstance(s[0], tp)]

    return statements

def all_statements(tree: ast.AST, tp: Optional[Type] = None) -> List[ast.AST]:
    """
    Return a list of all statements in `tree`.
    If `tp` is given, return only statements of that class.
    """

    return [stmt for stmt, func_name in all_statements_and_functions(tree, tp)]

class StatementMutator(NodeTransformer):
    """Mutate statements in an AST for automated repair."""

    def __init__(self,
                 suspiciousness_func:
                     Optional[Callable[[Tuple[Callable, int]], float]] = None,
                 source: Optional[List[ast.AST]] = None,
                 log: Union[bool, int] = False) -> None:
        """
        Constructor.
        `suspiciousness_func` is a function that takes a location
        (function, line_number) and returns a suspiciousness value
        between 0 and 1.0. If not given, all locations get the same 
        suspiciousness of 1.0.
        `source` is a list of statements to choose from.
        """

        super().__init__()
        self.log = log

        if suspiciousness_func is None:
            def suspiciousness_func(location: Tuple[Callable, int]) -> float:
                return 1.0
        assert suspiciousness_func is not None

        self.suspiciousness_func: Callable = suspiciousness_func

        if source is None:
            source = []
        self.source = source

        if self.log > 1:
            for i, node in enumerate(self.source):
                print(f"Source for repairs #{i}:")
                print_content(ast.unparse(node), '.py')
                print()
                print()

        self.mutations = 0
        

RE_SPACE = re.compile(r'[ \t\n]+')

class StatementMutator(StatementMutator):
    NODE_MAX_LENGTH = 20
    
    def node_suspiciousness(self, stmt: ast.AST, func_name: str) -> float:
        if not hasattr(stmt, 'lineno'):
            warnings.warn(f"{self.format_node(stmt)}: Expected line number")
            return 0.0

        suspiciousness = self.suspiciousness_func((func_name, stmt.lineno))
        if suspiciousness is None:  # not executed
            return 0.0

        return suspiciousness

    def format_node(self, node: ast.AST) -> str:
        """Return a string representation for `node`."""
        if node is None:
            return "None"

        if isinstance(node, list):
            return "; ".join(self.format_node(elem) for elem in node)

        s = RE_SPACE.sub(' ', ast.unparse(node)).strip()
        if len(s) > self.NODE_MAX_LENGTH - len("..."):
            s = s[:self.NODE_MAX_LENGTH] + "..."
        return repr(s)
        
    def node_to_be_mutated(self, tree: ast.AST) -> ast.AST:
        statements = all_statements_and_functions(tree)
        assert len(statements) > 0, "No statements"

        weights = [self.node_suspiciousness(stmt, func_name) 
                   for stmt, func_name in statements]
        stmts = [stmt for stmt, func_name in statements]

        if self.log > 1:
            print("Weights:")
            for i, stmt in enumerate(statements):
                node, func_name = stmt
                print(f"{weights[i]:.2} {self.format_node(node)}")

        if sum(weights) == 0.0:
            # No suspicious line
            return random.choice(stmts)
        else:
            return random.choices(stmts, weights=weights)[0]
        
    def choose_op(self) -> Callable:
        return random.choice([self.insert, self.swap, self.delete])

    def visit(self, node: ast.AST) -> ast.AST:
        super().visit(node)  # Visits (and transforms?) children

        if not node.mutate_me:  # type: ignore
            return node

        op = self.choose_op()
        new_node = op(node)
        self.mutations += 1

        if self.log:
            print(f"{node.lineno:4}:{op.__name__ + ':':7} "  # type: ignore
                  f"{self.format_node(node)} "
                  f"becomes {self.format_node(new_node)}")

        return new_node
    
    def choose_statement(self) -> ast.AST:
        return copy.deepcopy(random.choice(self.source))
    
    def swap(self, node: ast.AST) -> ast.AST:
        """Replace `node` with a random node from `source`"""
        new_node = self.choose_statement()

        if isinstance(new_node, ast.stmt):
            # The source `if P: X` is added as `if P: pass`
            if hasattr(new_node, 'body'):
                new_node.body = [ast.Pass()]  # type: ignore
            if hasattr(new_node, 'orelse'):
                new_node.orelse = []  # type: ignore
            if hasattr(new_node, 'finalbody'):
                new_node.finalbody = []  # type: ignore

        # ast.copy_location(new_node, node)
        return new_node
    
    def insert(self, node: ast.AST) -> Union[ast.AST, List[ast.AST]]:
        """Insert a random node from `source` after `node`"""
        new_node = self.choose_statement()

        if isinstance(new_node, ast.stmt) and hasattr(new_node, 'body'):
            # Inserting `if P: X` as `if P:`
            new_node.body = [node]  # type: ignore
            if hasattr(new_node, 'orelse'):
                new_node.orelse = []  # type: ignore
            if hasattr(new_node, 'finalbody'):
                new_node.finalbody = []  # type: ignore
            # ast.copy_location(new_node, node)
            return new_node

        # Only insert before `return`, not after it
        if isinstance(node, ast.Return):
            if isinstance(new_node, ast.Return):
                return new_node
            else:
                return [new_node, node]

        return [node, new_node]
    
    def delete(self, node: ast.AST) -> None:
        """Delete `node`."""

        branches = [attr for attr in ['body', 'orelse', 'finalbody']
                    if hasattr(node, attr) and getattr(node, attr)]
        if branches:
            # Replace `if P: S` by `S`
            branch = random.choice(branches)
            new_node = getattr(node, branch)
            return new_node

        if isinstance(node, ast.stmt):
            # Avoid empty bodies; make this a `pass` statement
            new_node = ast.Pass()
            ast.copy_location(new_node, node)
            return new_node

        return None  # Just delete
    
    def mutate(self, tree: ast.AST) -> ast.AST:
        """Mutate the given AST `tree` in place. Return mutated tree."""

        assert isinstance(tree, ast.AST)

        tree = copy.deepcopy(tree)

        if not self.source:
            self.source = all_statements(tree)

        for node in ast.walk(tree):
            node.mutate_me = False  # type: ignore

        node = self.node_to_be_mutated(tree)
        node.mutate_me = True  # type: ignore

        self.mutations = 0

        tree = self.visit(tree)

        if self.mutations == 0:
            warnings.warn("No mutations found")

        ast.fix_missing_locations(tree)
        return tree
    
class CrossoverOperator:
    """A class for performing statement crossover of Python programs"""

    def __init__(self, log: Union[bool, int] = False):
        """Constructor. If `log` is set, turn on logging."""
        self.log = log

    def cross_bodies(self, body_1: List[ast.AST], body_2: List[ast.AST]) -> \
        Tuple[List[ast.AST], List[ast.AST]]:
        """Crossover the statement lists `body_1` x `body_2`. Return new lists."""

        assert isinstance(body_1, list)
        assert isinstance(body_2, list)

        crossover_point_1 = len(body_1) // 2
        crossover_point_2 = len(body_2) // 2
        return (body_1[:crossover_point_1] + body_2[crossover_point_2:],
                body_2[:crossover_point_2] + body_1[crossover_point_1:])
        
class CrossoverOperator(CrossoverOperator):
    # In modules and class defs, the ordering of elements does not matter (much)
    SKIP_LIST = {ast.Module, ast.ClassDef}

    def can_cross(self, tree: ast.AST, body_attr: str = 'body') -> bool:
        if any(isinstance(tree, cls) for cls in self.SKIP_LIST):
            return False

        body = getattr(tree, body_attr, [])
        return body is not None and len(body) >= 2
    
    def crossover_attr(self, t1: ast.AST, t2: ast.AST, body_attr: str) -> bool:
        """
        Crossover the bodies `body_attr` of two trees `t1` and `t2`.
        Return True if successful.
        """
        assert isinstance(t1, ast.AST)
        assert isinstance(t2, ast.AST)
        assert isinstance(body_attr, str)

        if not getattr(t1, body_attr, None) or not getattr(t2, body_attr, None):
            return False

        if self.crossover_branches(t1, t2):
            return True

        if self.log > 1:
            print(f"Checking {t1}.{body_attr} x {t2}.{body_attr}")

        body_1 = getattr(t1, body_attr)
        body_2 = getattr(t2, body_attr)

        # If both trees have the attribute, we can cross their bodies
        if self.can_cross(t1, body_attr) and self.can_cross(t2, body_attr):
            if self.log:
                print(f"Crossing {t1}.{body_attr} x {t2}.{body_attr}")

            new_body_1, new_body_2 = self.cross_bodies(body_1, body_2)
            setattr(t1, body_attr, new_body_1)
            setattr(t2, body_attr, new_body_2)
            return True

        # Strategy 1: Find matches in class/function of same name
        for child_1 in body_1:
            if hasattr(child_1, 'name'):
                for child_2 in body_2:
                    if (hasattr(child_2, 'name') and
                           child_1.name == child_2.name):
                        if self.crossover_attr(child_1, child_2, body_attr):
                            return True

        # Strategy 2: Find matches anywhere
        for child_1 in random.sample(body_1, len(body_1)):
            for child_2 in random.sample(body_2, len(body_2)):
                if self.crossover_attr(child_1, child_2, body_attr):
                    return True

        return False
    
    def crossover_branches(self, t1: ast.AST, t2: ast.AST) -> bool:
        """Special case:
        `t1` = `if P: S1 else: S2` x `t2` = `if P': S1' else: S2'`
        becomes
        `t1` = `if P: S2' else: S1'` and `t2` = `if P': S2 else: S1`
        Returns True if successful.
        """
        assert isinstance(t1, ast.AST)
        assert isinstance(t2, ast.AST)

        if (hasattr(t1, 'body') and hasattr(t1, 'orelse') and
            hasattr(t2, 'body') and hasattr(t2, 'orelse')):

            t1 = cast(ast.If, t1)  # keep mypy happy
            t2 = cast(ast.If, t2)

            if self.log:
                print(f"Crossing branches {t1} x {t2}")

            t1.body, t1.orelse, t2.body, t2.orelse = \
                t2.orelse, t2.body, t1.orelse, t1.body
            return True

        return False
    
    def crossover(self, t1: ast.AST, t2: ast.AST) -> Tuple[ast.AST, ast.AST]:
        """Do a crossover of ASTs `t1` and `t2`.
        Raises `CrossoverError` if no crossover is found."""
        assert isinstance(t1, ast.AST)
        assert isinstance(t2, ast.AST)

        for body_attr in ['body', 'orelse', 'finalbody']:
            if self.crossover_attr(t1, t2, body_attr):
                return t1, t2

        raise CrossoverError("No crossover found")
    
class CrossoverError(ValueError):
    pass
    

class Repairer(StackInspector):
    """A class for automatic repair of Python programs"""

    def __init__(self, debugger: RankingDebugger, *,
                 targets: Optional[List[Any]] = None,
                 sources: Optional[List[Any]] = None,
                 log: Union[bool, int] = False,
                 mutator_class: Type = StatementMutator,
                 crossover_class: Type = CrossoverOperator,
                 reducer_class: Type = DeltaDebugger,
                 globals: Optional[Dict[str, Any]] = None):
        """Constructor.
`debugger`: a `RankingDebugger` to take tests and coverage from.
`targets`: a list of functions/modules to be repaired.
    (default: the covered functions in `debugger`, except tests)
`sources`: a list of functions/modules to take repairs from.
    (default: same as `targets`)
`globals`: if given, a `globals()` dict for executing targets
    (default: `globals()` of caller)"""

        assert isinstance(debugger, RankingDebugger)
        self.debugger = debugger
        self.log = log

        if targets is None:
            targets = self.default_functions()
        if not targets:
            raise ValueError("No targets to repair")

        if sources is None:
            sources = self.default_functions()
        if not sources:
            raise ValueError("No sources to take repairs from")

        if self.debugger.function() is None:
            raise ValueError("Multiple entry points observed")

        self.target_tree: ast.AST = self.parse(targets)
        self.source_tree: ast.AST = self.parse(sources)

        self.log_tree("Target code to be repaired:", self.target_tree)
        if ast.dump(self.target_tree) != ast.dump(self.source_tree):
            self.log_tree("Source code to take repairs from:", 
                          self.source_tree)

        self.fitness_cache: Dict[str, float] = {}

        self.mutator: StatementMutator = \
            mutator_class(
                source=all_statements(self.source_tree),
                suspiciousness_func=self.debugger.suspiciousness,
                log=(self.log >= 3))
        self.crossover: CrossoverOperator = crossover_class(log=(self.log >= 3))
        self.reducer: DeltaDebugger = reducer_class(log=(self.log >= 3))

        if globals is None:
            globals = self.caller_globals()  # see below

        self.globals = globals
        
class Repairer(Repairer):
    def getsource(self, item: Union[str, Any]) -> str:
        """Get the source for `item`. Can also be a string."""

        if isinstance(item, str):
            item = self.globals[item]
        return inspect.getsource(item)
    
    def default_functions(self) -> List[Callable]:
        """Return the set of functions to be repaired.
        Functions whose names start or end in `test` are excluded."""
        def is_test(name: str) -> bool:
            return name.startswith('test') or name.endswith('test')

        return [func for func in self.debugger.covered_functions()
                if not is_test(func.__name__)]
        
    def log_tree(self, description: str, tree: Any) -> None:
        """Print out `tree` as source code prefixed by `description`."""
        if self.log:
            print(description)
            print_content(ast.unparse(tree), '.py')
            print()
            print()
    
    def parse(self, items: List[Any]) -> ast.AST:
        """Read in a list of items into a single tree"""
        tree = ast.parse("")
        for item in items:
            if isinstance(item, str):
                item = self.globals[item]

            item_lines, item_first_lineno = inspect.getsourcelines(item)

            try:
                item_tree = ast.parse("".join(item_lines))
            except IndentationError:
                # inner function or likewise
                warnings.warn(f"Can't parse {item.__name__}")
                continue

            ast.increment_lineno(item_tree, item_first_lineno - 1)
            tree.body += item_tree.body

        return tree
    
    def run_test_set(self, test_set: str, validate: bool = False) -> int:
        """
        Run given `test_set`
        (`DifferenceDebugger.PASS` or `DifferenceDebugger.FAIL`).
        If `validate` is set, check expectations.
        Return number of passed tests.
        """
        passed = 0
        collectors = self.debugger.collectors[test_set]
        function = self.debugger.function()
        assert function is not None
        # FIXME: function may have been redefined

        for c in collectors:
            if self.log >= 4:
                print(f"Testing {c.id()}...", end="")

            try:
                function(**c.args())
            except Exception as err:
                if self.log >= 4:
                    print(f"failed ({err.__class__.__name__})")

                if validate and test_set == self.debugger.PASS:
                    raise err.__class__(
                        f"{c.id()} should have passed, but failed")
                continue

            passed += 1
            if self.log >= 4:
                print("passed")

            if validate and test_set == self.debugger.FAIL:
                raise FailureNotReproducedError(
                    f"{c.id()} should have failed, but passed")

        return passed
    
    def weight(self, test_set: str) -> float:
        """
        Return the weight of `test_set`
        (`DifferenceDebugger.PASS` or `DifferenceDebugger.FAIL`).
        """
        return {
            self.debugger.PASS: WEIGHT_PASSING,
            self.debugger.FAIL: WEIGHT_FAILING
        }[test_set]

    def run_tests(self, validate: bool = False) -> float:
        """Run passing and failing tests, returning weighted fitness."""
        fitness = 0.0

        for test_set in [self.debugger.PASS, self.debugger.FAIL]:
            passed = self.run_test_set(test_set, validate=validate)
            ratio = passed / len(self.debugger.collectors[test_set])
            fitness += self.weight(test_set) * ratio

        return fitness
    
    def validate(self) -> None:
        fitness = self.run_tests(validate=True)
        assert fitness == self.weight(self.debugger.PASS)
        
    def fitness(self, tree: ast.AST) -> float:
        """Test `tree`, returning its fitness"""
        key = cast(str, ast.dump(tree))
        if key in self.fitness_cache:
            return self.fitness_cache[key]

        # Save defs
        original_defs: Dict[str, Any] = {}
        for name in self.toplevel_defs(tree):
            if name in self.globals:
                original_defs[name] = self.globals[name]
            else:
                warnings.warn(f"Couldn't find definition of {repr(name)}")

        assert original_defs, f"Couldn't find any definition"

        if self.log >= 3:
            print("Repair candidate:")
            print_content(ast.unparse(tree), '.py')
            print()

        # Create new definition
        try:
            code = compile(cast(ast.Module, tree), '<Repairer>', 'exec')
        except ValueError:  # Compilation error
            code = None

        if code is None:
            if self.log >= 3:
                print(f"Fitness = 0.0 (compilation error)")

            fitness = 0.0
            return fitness

        # Execute new code, defining new functions in `self.globals`
        exec(code, self.globals)

        # Set new definitions in the namespace (`__globals__`)
        # of the function we will be calling.
        function = self.debugger.function()
        assert function is not None
        assert hasattr(function, '__globals__')

        for name in original_defs:
            function.__globals__[name] = self.globals[name]  # type: ignore

        fitness = self.run_tests(validate=False)

        # Restore definitions
        for name in original_defs:
            function.__globals__[name] = original_defs[name]  # type: ignore
            self.globals[name] = original_defs[name]

        if self.log >= 3:
            print(f"Fitness = {fitness}")

        self.fitness_cache[key] = fitness
        return fitness
    
    def toplevel_defs(self, tree: ast.AST) -> List[str]:
        """Return a list of names of defined functions and classes in `tree`"""
        visitor = DefinitionVisitor()
        visitor.visit(tree)
        assert hasattr(visitor, 'definitions')
        return visitor.definitions
    
    def initial_population(self, size: int) -> List[ast.AST]:
        """Return an initial population of size `size`"""
        return [self.target_tree] + \
            [self.mutator.mutate(copy.deepcopy(self.target_tree))
                for i in range(size - 1)]

    def repair(self, population_size: int = POPULATION_SIZE, iterations: int = 100) -> \
        Tuple[ast.AST, float]:
        """
        Repair the function we collected test runs from.
        Use a population size of `population_size` and
        at most `iterations` iterations.
        Returns a pair (`ast`, `fitness`) where 
        `ast` is the AST of the repaired function, and
        `fitness` is its fitness (between 0 and 1.0)
        """
        self.validate()

        population = self.initial_population(population_size)

        last_key = ast.dump(self.target_tree)

        for iteration in range(iterations):
            population = self.evolve(population)

            best_tree = population[0]
            fitness = self.fitness(best_tree)

            if self.log:
                print(f"Evolving population: "
                      f"iteration{iteration:4}/{iterations} "
                      f"fitness = {fitness:.5}   \r", end="")

            if self.log >= 2:
                best_key = ast.dump(best_tree)
                if best_key != last_key:
                    print()
                    print()
                    self.log_tree(f"New best code (fitness = {fitness}):",
                                  best_tree)
                    last_key = best_key

            if fitness >= 1.0:
                break

        if self.log:
            print()

        if self.log and self.log < 2:
            self.log_tree(f"Best code (fitness = {fitness}):", best_tree)

        best_tree = self.reduce(best_tree)
        fitness = self.fitness(best_tree)

        self.log_tree(f"Reduced code (fitness = {fitness}):", best_tree)

        return best_tree, fitness
    
    def evolve(self, population: List[ast.AST]) -> List[ast.AST]:
        """Evolve the candidate population by mutating and crossover."""
        n = len(population)

        # Create offspring as crossover of parents
        offspring: List[ast.AST] = []
        while len(offspring) < n:
            parent_1 = copy.deepcopy(random.choice(population))
            parent_2 = copy.deepcopy(random.choice(population))
            try:
                self.crossover.crossover(parent_1, parent_2)
            except CrossoverError:
                pass  # Just keep parents
            offspring += [parent_1, parent_2]

        # Mutate offspring
        offspring = [self.mutator.mutate(tree) for tree in offspring]

        # Add it to population
        population += offspring

        # Keep the fitter part of the population
        population.sort(key=self.fitness_key, reverse=True)
        population = population[:n]

        return population
    
    def fitness_key(self, tree: ast.AST) -> Tuple[float, int]:
        """Key to be used for sorting the population"""
        tree_size = len([node for node in ast.walk(tree)])
        return (self.fitness(tree), -tree_size)
    
    def reduce(self, tree: ast.AST) -> ast.AST:
        """Simplify `tree` using delta debugging."""

        original_fitness = self.fitness(tree)
        source_lines = ast.unparse(tree).split('\n')

        with self.reducer:
            self.test_reduce(source_lines, original_fitness)

        reduced_lines = self.reducer.min_args()['source_lines']
        reduced_source = "\n".join(reduced_lines)

        return ast.parse(reduced_source)
    
    def test_reduce(self, source_lines: List[str], original_fitness: float) -> None:
        """Test function for delta debugging."""

        try:
            source = "\n".join(source_lines)
            tree = ast.parse(source)
            fitness = self.fitness(tree)
            assert fitness < original_fitness

        except AssertionError:
            raise
        except SyntaxError:
            raise
        except IndentationError:
            raise
        except Exception:
            # traceback.print_exc()  # Uncomment to see internal errors
            raise

    
    
class FailureNotReproducedError(ValueError):
    pass

class DefinitionVisitor(NodeVisitor):
    def __init__(self) -> None:
        self.definitions: List[str] = []

    def add_definition(self, node: Union[ast.ClassDef, 
                                         ast.FunctionDef, 
                                         ast.AsyncFunctionDef]) -> None:
        self.definitions.append(node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.add_definition(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.add_definition(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.add_definition(node)


def all_conditions(trees: Union[ast.AST, List[ast.AST]],
                   tp: Optional[Type] = None) -> List[ast.expr]:
    """
    Return all conditions from the AST (or AST list) `trees`.
    If `tp` is given, return only elements of that type.
    """

    if not isinstance(trees, list):
        assert isinstance(trees, ast.AST)
        trees = [trees]

    visitor = ConditionVisitor()
    for tree in trees:
        visitor.visit(tree)
    conditions = visitor.conditions
    if tp is not None:
        conditions = [c for c in conditions if isinstance(c, tp)]

    return conditions

class ConditionVisitor(NodeVisitor):
    def __init__(self) -> None:
        self.conditions: List[ast.expr] = []
        self.conditions_seen: Set[str] = set()
        super().__init__()

    def add_conditions(self, node: ast.AST, attr: str) -> None:
        elems = getattr(node, attr, [])
        if not isinstance(elems, list):
            elems = [elems]

        elems = cast(List[ast.expr], elems)

        for elem in elems:
            elem_str = ast.unparse(elem)
            if elem_str not in self.conditions_seen:
                self.conditions.append(elem)
                self.conditions_seen.add(elem_str)

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.add_conditions(node, 'values')
        return super().generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        if isinstance(node.op, ast.Not):
            self.add_conditions(node, 'operand')
        return super().generic_visit(node)

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if hasattr(node, 'test'):
            self.add_conditions(node, 'test')
        return super().generic_visit(node)
    

class ConditionMutator(StatementMutator):
    """Mutate conditions in an AST"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Constructor. Arguments are as with `StatementMutator` constructor."""
        super().__init__(*args, **kwargs)
        self.conditions = all_conditions(self.source)
        if self.log:
            print("Found conditions",
                  [ast.unparse(cond).strip() 
                   for cond in self.conditions])

    def choose_condition(self) -> ast.expr:
        """Return a random condition from source."""
        return copy.deepcopy(random.choice(self.conditions))
    
class ConditionMutator(ConditionMutator):
    def choose_bool_op(self) -> str:
        return random.choice(['set', 'not', 'and', 'or'])

    def swap(self, node: ast.AST) -> ast.AST:
        """Replace `node` condition by a condition from `source`"""
        if not hasattr(node, 'test'):
            return super().swap(node)

        node = cast(ast.If, node)

        cond = self.choose_condition()
        new_test = None

        choice = self.choose_bool_op()

        if choice == 'set':
            new_test = cond
        elif choice == 'not':
            new_test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        elif choice == 'and':
            new_test = ast.BoolOp(op=ast.And(), values=[cond, node.test])
        elif choice == 'or':
            new_test = ast.BoolOp(op=ast.Or(), values=[cond, node.test])
        else:
            raise ValueError("Unknown boolean operand")

        if new_test:
            # ast.copy_location(new_test, node)
            node.test = new_test

        return node