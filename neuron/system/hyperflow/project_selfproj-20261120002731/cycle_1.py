def analyze_code_snippet(code_snippet):
    # Criteria for grading
    # 1. Docstring presence
    # 2. Proper indentation
    # 3. Code length (simple heuristic: less than 50 lines)
    
    # Check for docstring
    if not code_snippet.strip().startswith('"""') and not code_snippet.strip().startswith("'''"):
        docstring_status = "No docstring"
    else:
        docstring_status = "Docstring present"
    
    # Check for proper indentation (simple heuristic: no tabs, at least one space)
    lines = code_snippet.split('\n')
    indentation_status = "Proper indentation"
    for line in lines:
        if line and not line.isspace() and line[0] in [' ', '\t']:
            if line[0] == '\t':
                indentation_status = "Improper indentation (tabs used)"
                break
    
    # Check for code length
    code_length_status = "Code length is acceptable"
    if len(code_snippet.split('\n')) > 50:
        code_length_status = "Code length is too long"
    
    # Calculate grade based on criteria
    grade = 0
    if docstring_status == "Docstring present":
        grade += 1
    if indentation_status == "Proper indentation":
        grade += 1
    if code_length_status == "Code length is acceptable":
        grade += 1
    
    # Print grade and feedback
    print(f"Grade: {grade}/3")
    print(f"Feedback: {docstring_status}, {indentation_status}, {code_length_status}")
    
    return grade

# Test cases
assert analyze_code_snippet('"""This is a docstring"""') == 3
assert analyze_code_snippet('def test():\n    pass') == 3
assert analyze_code_snippet('def test():\n\tdef inner():\n\t    pass') == 1
assert analyze_code_snippet('def test():\n    pass\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n') == 2

print('PASS')