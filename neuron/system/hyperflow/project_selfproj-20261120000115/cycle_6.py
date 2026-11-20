def generate_code_snippet(template, existing_code):
    # Example template: "def func_name(param1, param2): return param1 + param2"
    # Example existing_code: "def add(a, b): return a + b"
    generated_snippet = template.format(**locals())
    return generated_snippet

# Test the function
assert generate_code_snippet("def func_name(param1, param2): return param1 + param2", "def add(a, b): return a + b") == "def func_name(param1, param2): return param1 + param2"
print('PASS')