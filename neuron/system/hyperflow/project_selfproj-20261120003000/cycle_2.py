def analyze_code(code):
    lines = code.split('\n')
    issues = []

    # Check for formatting issues
    if any(line.strip() == '' for line in lines):
        issues.append('Formatting: Empty lines detected')

    # Check for redundant code
    if '    ' in code and 'def ' in code:
        issues.append('Redundant: Indentation with four spaces is common, but check for redundant indentation')

    # Check for unused code
    if 'unused_code' in code:
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