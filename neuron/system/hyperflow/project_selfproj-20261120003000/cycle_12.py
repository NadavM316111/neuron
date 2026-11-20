import ast

def detect_unused(src, tree):
    used_names = set()
    unused_imports = set()
    import_nodes = []

    # Collect all used names and import nodes
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                unused_imports.add(alias.name)
                import_nodes.append(node)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                unused_imports.add(f"{node.module}.{alias.name}")
                import_nodes.append(node)

    # Collect all names assigned in the code
    assigned_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_names.add(target.id)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                assigned_names.add(node.target.id)

    # Identify unused variables
    unused_variables = assigned_names - used_names

    # Identify unused imports
    for node in import_nodes:
        for alias in node.names:
            if f"{node.module}.{alias.name}" in unused_variables or alias.name in unused_variables:
                unused_imports.discard(f"{node.module}.{alias.name}" if node.module else alias.name)

    # Prepare the list of unused issues
    unused_issues = []
    for name in unused_variables:
        unused_issues.append(f"Unused variable: {name}")
    for imp in unused_imports:
        unused_issues.append(f"Unused import: {imp}")

    return unused_issues
def detect_redundant(src, tree):
    assignments = {}
    redundant_issues = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id in assignments:
                        redundant_issues.append(f"Line {node.lineno}: Redundant assignment to {target.id}")
                    else:
                        assignments[target.id] = node.value
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                assignments[node.target.id] = node.value

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if isinstance(node.value, ast.BinOp) and isinstance(node.value.left, ast.Name) and node.value.left.id in assignments:
                        if assignments[node.value.left.id] == node.value.right:
                            redundant_issues.append(f"Line {node.lineno}: Redundant computation in assignment to {target.id}")

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
