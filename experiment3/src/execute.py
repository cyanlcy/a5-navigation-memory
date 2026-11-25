"""Single-question Python worker. Receives only public data. Nothing is written by candidate code."""
import ast
import builtins
import json
import math
import resource
import sys
import time

SAFE = {key: getattr(builtins, key) for key in
        "abs all any bool dict enumerate float int len list max min range round sorted sum tuple zip str".split()}
SAFE.update({key: getattr(math,key) for key in ("sin","cos","tan","atan","atan2","radians","sqrt")})
NODES = set("Module FunctionDef arguments arg Assign AugAssign Return Expr If For Break Continue Pass "
            "Name Store Load Tuple List Dict Constant Subscript Slice Call keyword Attribute Compare "
            "Gt Lt GtE LtE Eq NotEq In NotIn Is IsNot BoolOp And Or BinOp Add Sub Mult Div FloorDiv Mod "
            "UnaryOp USub UAdd Not IfExp ListComp comprehension GeneratorExp".split())
METHODS = {"match", "identify", "direct", "locate", "point"}
def compile_source(source):
    tree = ast.parse(source)
    if len(source) > 10000 or any(not isinstance(node, ast.FunctionDef) for node in tree.body):
        raise ValueError("submit only function definitions")
    names = {node.name for node in tree.body}
    if "decide" not in names or names & (set(SAFE) | {"tools"}):
        raise ValueError("invalid function name")
    for node in ast.walk(tree):
        if type(node).__name__ not in NODES:
            raise ValueError("unsupported Python construct " + type(node).__name__)
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "tools" and node.attr in METHODS):
                raise ValueError("only declared visual tool attributes are available")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Attribute) or
                isinstance(node.func, ast.Name) and node.func.id in names | set(SAFE)):
            raise ValueError("unknown call")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("private name")
    return compile(tree, "<decision>", "exec")

class VisualTools:
    def __init__(self, values):
        self.values = values
        self.calls = []
        self.attempts = []

    def fetch(self, name):
        self.attempts.append(name)
        if len(self.calls) >= 2:
            raise ValueError("visual budget exceeded")
        self.calls.append(name)
        return self.values[name]

    def match(self): return self.fetch("match")
    def identify(self): return self.fetch("identify")
    def direct(self): return self.fetch("direct")
    def locate(self): return self.fetch("locate")
    def point(self): return self.fetch("point")

def run(source, row):
    tools = VisualTools(row["visual"])
    start = time.monotonic()
    try:
        scope = {"__builtins__": SAFE}
        exec(compile_source(source), scope)
        frame, action, trace = scope["decide"](row["observation"], tools)
        if type(frame) is not int or frame not in (0, 1) or action not in ("left", "right") or not isinstance(trace, dict):
            raise ValueError("expected (frame 0/1, left/right, trace dict)")
        if len(json.dumps(trace, allow_nan=False)) > 2048:
            raise ValueError("invalid trace")
        result = dict(frame=frame, action=action, trace=trace)
    except Exception as error:
        result = dict(frame=None, action=None, error=(type(error).__name__ + ": " + str(error))[:160])
    return dict(**result, calls=tools.calls, attempts=tools.attempts, seconds=time.monotonic()-start)

if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
    job = json.load(sys.stdin)
    isolated="--seal" in sys.argv
    if isolated:
        # -I removes the script path; load only our trusted sibling before chroot.
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from isolation import single_question
        single_question()
    result=run(job["source"],job["row"])
    print(json.dumps(dict(**result,isolated=isolated), allow_nan=False))
