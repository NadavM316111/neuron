# DRAFT contribution (PREPARED, NOT SUBMITTED - gated)

repo: NadavM316111/neuron
issue: Add an is_palindrome(s) helper
url: https://github.com/NadavM316111/neuron/issues/2

**PLAN:**  
1. Define the function `is_palindrome(s)`.
2. Normalize the string by converting it to lowercase and removing non-alphanumeric characters.
3. Check if the normalized string is equal to its reverse.

**CODE:**  
```python
def is_palindrome(s):
    normalized = ''.join(c.lower() for c in s if c.isalnum())
    return normalized == normalized[::-1]
```

**PR_TITLE:**  
Add is_palindrome(s) helper function

**PR_BODY:**  
This PR adds a helper function `is_palindrome(s)` that checks if a given string is a palindrome, ignoring case and non-alphanumeric characters. The function first normalizes the string by converting it to lowercase and removing any non-alphanumeric characters, then checks if the normalized string is equal to its reverse.
