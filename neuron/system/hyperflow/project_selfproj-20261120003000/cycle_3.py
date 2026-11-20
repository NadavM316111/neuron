def analyze_code(code):
    lines = code.split('\n')
    issues = []

    # Check for formatting issues
    if any(line.strip() == '' for line in lines):
        issues.append('Formatting: Empty lines detected')

    # Check for redundant code
    indent_level = 0
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('def ') or stripped_line.startswith('class '):
            indent_level = len(line) - len(stripped_line)
        elif '    ' in line and indent_level == 4:
            issues.append('Redundant: Indentation with four spaces is common, but check for redundant indentation')

    # Check for unused code
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('def ') or stripped_line.startswith('class '):
            indent_level = len(line) - len(stripped_line)
        elif 'unused_code' in line and not any(keyword in line for keyword in ['return', 'yield', 'if', 'for', 'while']):
            issues.append('Unused: The string "unused_code" is present but not used')

    return issues

def main():
    code = """
def example_function(x, y):
    z = x + y
    return z
    unused_code = 0  # This line is not used
"""
    report = analyze_code(code)
    if report:
        print(f"Detected issues: {', '.join(report)}")
    else:
        print("Code is clean")

if __name__ == "__main__":
    main()