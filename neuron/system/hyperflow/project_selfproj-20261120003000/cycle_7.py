import ast

def detect_unused(src, tree):
    return []

def detect_redundant(src, tree):
    assignments = {}
    redundant_issues = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    value = ast.unparse(node.value)
                    if target.id in assignments:
                        if assignments[target.id] != value:
                            redundant_issues.append(f"Redundant assignment on line {node.lineno}: {target.id} assigned to different values.")
                    else:
                        assignments[target.id] = value
        elif isinstance(node, ast.AugAssign):
            # Ignore augmented assignments
            continue

    # Check for repeated computations
    computations = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) or isinstance(node, ast.UnaryOp):
            value = ast.unparse(node)
            if value in computations:
                redundant_issues.append(f"Redundant computation on line {node.lineno}: {value} computed multiple times.")
            else:
                computations[value] = node.lineno

    return redundant_issues
def detect_formatting(src, tree):
    return []


def analyze_code(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "syntax error"
    issues = []
    for d in (detect_unused, detect_redundant, detect_formatting):
        try:
            issues += list(d(src, tree) or [])
        except Exception:
            pass
    return "; ".join(dict.fromkeys(str(i).lower() for i in issues)) if issues else "clean, no issues"
print(analyze_code("x=1"))
